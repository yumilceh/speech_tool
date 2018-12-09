import pyaudio
import wave
from array import array
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math
from scipy.signal import lfilter
from scikits.talkbox import lpc
import time
import matplotlib.pyplot as plt
from divapy import Diva as Divapy
import os
from sklearn import preprocessing
import pandas as pd
import skimage.measure
from scipy import misc
from collections import Iterable
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm

max_F0 = 200.

#record sound to wav file
def record_sound(file_name, time, rate=11025, FORMAT=pyaudio.paInt16, CHANNELS=1, chunk=1024):
    #Press  key for start recording
    raw_input("Press Enter to start recording...")

    audio=pyaudio.PyAudio() #instantiate the pyaudio

    #recording prerequisites
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=rate, input=True, frames_per_buffer=chunk)

    #starting recording
    frames=[]

    for i in range(0, int(rate / chunk * time)):
        data=stream.read(chunk)
        data_chunk=array('h',data)
        vol=max(data_chunk)
        if(vol>=500):
            print("something said")
            frames.append(data)
        else:
            print("nothing")
        print("\n")

    #end of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #writing to file
    wavfile_ = wave.open(file_name, 'wb')
    wavfile_.setnchannels(CHANNELS)
    wavfile_.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile_.setframerate(rate)
    wavfile_.writeframes(b''.join(frames))#append frames recorded to file
    wavfile_.close()

#load wav file and retun sound signal and Fs of the sound
def load_sound(file_name):
    fs, sound = wavfile.read(file_name)
    return sound

#plaing sound from sound signal
def play_sound(sound):  # keep in mind that DivaMatlab works with ts=0.005
    pa = pyaudio.PyAudio()  # If pa and stream are not elements of the self object then sound does not play
    stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                          rate=11025,
                          output=True)
    stream.start_stream()
    stream.write(sound.astype(np.int16).tostring())
    time.sleep(len(sound)/11025. + 0.2)

#return formants frequencies(f0,f1,f2,f3,...,fn) from sound signal
def get_fromants(sound, fs=11025):
    # Read from file.
    # spf = sound

    x = sound
    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1.], [1., 0.63], x1)

    # Get LPC.
    ncoeff = 2 + fs / 1000
    A, e, k = lpc(x1, ncoeff)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Fs = spf.getframerate() #Gregory comment
    frqs = sorted(angz * (fs / (2 * math.pi)))

    return frqs

#plots fromants freq
def plot_formants(formants, labels = None):
    i = 0
    for formant in formants:
        #print(formant)
        plt.plot(formant[1], formant[2], 'ob')
        plt.annotate(labels[i], xy=(formant[1], formant[2]))
        i += 1

#record volwes from created sound signal
def record_vt_sound(art,time, fs = 11025):
    diva_synth = Divapy()
    n_arts = round((time/0.005))+1
    arts = np.tile(art, (int(n_arts), 1))
    sound = diva_synth.get_sound(arts)
    scaled = np.int16(sound / np.max(np.abs(sound)) * 32767)
    wavfile.write("vt" + '.wav', 11025, scaled)
    diva_synth.play_sound(sound)

#create german wolves to wav file from german_arts
def create_german_files(german_art, time, fs=11025):
    diva_synth = Divapy()
    n_arts = round((time/0.005))+1
    vow_names = ['-E', '-E_', '-I', '-O', '-U', '-Y', '-Z_', '-a', '-a_', '-at', '-b', '-e_', '-i_', '-o_', '-p', '-u_', '-y']
    for i in range(len(german_art[:, 0])):
        german_vowels = german_art[i, :]
        german_vowels = np.tile(german_vowels, (int(n_arts), 1))
        german_sound = diva_synth.get_sound(german_vowels)
        scaled = np.int16(german_sound / np.max(np.abs(german_sound)) * 32767)
        wavfile.write("vt" + vow_names[i] + '.wav', fs, scaled)
    #diva_synth.play_sound(german_sound)

#### ---> INT TO FLOAT AND FLOAT TO INT <--- ####
def int_to_float(data):
    float_data = map(float, data)
    # float_data = np.memmap(data, dtype='float32')
    return float_data

def float_to_int(data):
    int_data = map(int, data)
    # int_data = np.memmap(data, dtype='int16')
    return int_data

#recording voice to folder and then to .wav file (better one)
def record_sound_save_to_folder(file_name, time, directory, rate=11025, FORMAT=pyaudio.paInt16, CHANNELS=1, chunk=1024):
    #Press  key for start recording
    raw_input("Press Enter to start recording...")
    audio=pyaudio.PyAudio() #instantiate the pyaudio
    #recording prerequisites
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=rate, input=True, frames_per_buffer=chunk)
    #starting recording
    frames=[]
    for i in range(0, int(rate / chunk * time)):
        data=stream.read(chunk)
        data_chunk=array('h',data)
        vol=max(data_chunk)
        if(vol>=500):
            print("something said")
            frames.append(data)
        else:
            print("nothing")
        print("\n")

    #end of recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #writing to file
    directory = directory
    completeName = os.path.join(directory, file_name)
    wavfile_ = wave.open(completeName, 'wb')
    wavfile_.setnchannels(CHANNELS)
    wavfile_.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile_.setframerate(rate)
    wavfile_.writeframes(b''.join(frames))#append frames recorded to file
    wavfile_.close()

#part of recording file
def creat_base_of_files(list_of_words, t):
    person = raw_input('Enter your name: ')
    print('Hello ' + person)
    #creating folder with name of the person
    directory = './'+person+'/'
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

    #creating list of name (name_word_1_iteration_2.wav)
    files = []
    for i in list_of_words:
        files.append(person + "_word_" + i + "_iteration_")
    #record sound and save in file
    for i, file__ in enumerate(files):
        for j in range(1, 3):
            file_ = file__ + str(j) + ".wav"
            print("Next sound " + list_of_words[i])
            record_sound_save_to_folder(file_ , t, directory)
            print("Created sound " + str(file_))

            completeName = os.path.join(directory, file_)
            print(completeName)
            sound = load_sound(completeName)
            play_sound(sound)

            answer = raw_input("Sound its correct? Y/N\n")
            while answer != 'y':
                os.remove(completeName)
                print("Repeating sound " + str(file_))
                record_sound_save_to_folder(file_, t, directory)
                sound = load_sound(completeName)
                play_sound(sound)
                print("Created sound " + str(file_))
                answer = raw_input("Sound its correct? Y/N\n")
    return files, directory

# get chunks of the sound and return list of parting sound in arguments we are giving t_sw (time sieze window)
# and t_ol (time overloping) in ms

def get_chunks(sound, t_sw=50, t_ol=25, fs = 11025): #time in miliseconds
    width_size = int(math.ceil((t_sw/1000.)*fs) + 1)
    width_overlaping = int(math.ceil((t_ol/1000.)*fs) + 1)
    size_of_sound = len(sound)
    step_move = width_size - width_overlaping
    list_of_chunks=[]
    for j in range(0, size_of_sound - (width_size - width_overlaping), step_move):
        sub_samples = sound[j:width_size]
        width_size = width_size + step_move
        list_of_chunks.append([sub_samples])
    return list_of_chunks


def get_formants_trayectory(list_of_chunks):
    chunks_formants = [get_fromants(chunk[0]) for chunk in list_of_chunks]
    for i in range(len(chunks_formants)):
        created_formants_array = np.array(chunks_formants[i])
        non_zero_index = np.nonzero(created_formants_array)[0][0]
        # print (non_zero_index)
        if created_formants_array[non_zero_index] > max_F0:
            chunks_formants[i] = [0.] + list(created_formants_array[non_zero_index:non_zero_index+4])
        else:
            chunks_formants[i] = list(created_formants_array[non_zero_index:non_zero_index+5])
    return chunks_formants

# scaing chunks of our formants
def get_scaling(chunks_formants):
    x = preprocessing.MinMaxScaler().fit_transform(chunks_formants)
    normalization_data = pd.DataFrame(x)
    return normalization_data

# prepering image to model, Conv, Rel and Pooling making image more comprested but with one kernel
def compare_kernels(image, *args):
    for arg in args:
        temp_image = misc.np.array(image).T
        Conv = signal.convolve2d(temp_image, arg)
        ReLu = np.maximum(Conv, 0)
        reduced_image = skimage.measure.block_reduce(ReLu, (2, 2), np.max)
        plt.figure()
        plt.imshow(reduced_image.T, cmap='gray')

    plt.show()
    return reduced_image

# doing the same like compare_kernels but here we are giving list of kernels and result is image after all kernels
def apply_conv(image, *args):
    temp_image = misc.np.array(image).T
    # plt.subplot(len(args)+1, 1, 1)
    # plt.imshow(image, cmap='gray')
    for i, arg in enumerate(args):
        Conv = signal.convolve2d(temp_image, arg)
        ReLu = np.maximum(Conv, 0)
        temp_image = skimage.measure.block_reduce(ReLu, (2, 2), np.max)

    #     plt.subplot(len(args)+1, 1, i + 2)
    #     plt.imshow(temp_image.T, cmap='gray')
    #
    # plt.show()
    return temp_image

#create one-dimensional list from many dimensional list ([1,1],[2,2]) = ([1,1,2,2])
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, basestring):
            for x in flatten(item):
                yield x
        else:
            yield item

def chage_namefile_to_number(filename):
    words = ['hello', 'who', 'which', 'where', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'one', 'two', 'three',
             'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'red', 'blue', 'green', 'white', 'black', 'orange',
             'pink', 'yellow', 'hours', 'duck', 'pig', 'cow', 'chicken', 'dog', 'cat', 'mouse', 'rat', 'bee', 'bat',
             'lion', 'house', 'block', 'flat', 'line', 'square', 'triangle', 'circle', 'cube']

    for i, word in enumerate(words):
        if '_'+word+'_' in filename:
            # print(i)
            return i

#prepering data, creating list of pix(firt value), numer of world(second value)
def creating_list_of_pixels():
    name = raw_input("Put name folder: ")
    directory = './' + name + '/'
    temp_files = os.listdir(directory)
    pix_list = []
    for i in temp_files:
        temp_name = directory + str(i)
        # print (temp_name)
        loaded_files = load_sound(temp_name)
        chunks_list = get_chunks(loaded_files)
        chunk_formants = get_formants_trayectory(chunks_list)
        # normalization = get_scaling(chunk_formants)
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(chunk_formants)
        scaled_data = scaler.transform(chunk_formants)
        # scaled_val_data = scaler.transform()
        edge_detection = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        sharpen_image = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        list_krenels2 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        # reduced_image = compare_kernels(np.array(normalization).T, edge_detection, sharpen_image, list_krenels2)
        new_image = apply_conv(np.array(scaled_data).T, edge_detection, sharpen_image, list_krenels2)
        ####
        # plt.imshow(new_image.T, cmap='gray')

        temp_list = []
        pix_ = (list(flatten(new_image.T)))

        ####### np.flaot to float
        # print(pix_)
        # print(type(pix_[0]))
        # pix_ = [float(j) for j in pix_]
        # print(pix_)
        # print(type(pix_[0]))

        temp_list.extend(pix_) #####

        # print (i)
        word_numer = chage_namefile_to_number(i)
        temp_list.append(word_numer)
        # print (temp_list)
        pix_list.append(temp_list)
        # print (pix_list)
        # plt.savefig(temp_name + '.png')
        # plt.show()
    return pix_list

#open all files and check the length which is longest :keepmax
#open file by file
#add 0 to the end or begin of the sound
from shutil import copyfile
def same_file_length(pix_list):
    x = 0
    for pix_element in pix_list:
        # print("przed : \n ", pix_element)
        if len(pix_element) < 34:
            counter = 34 - len(pix_element)
            # half = counter / 2
            # rest = counter % 2
            # print(counter)
            for i in range(counter):
                # pix_element[0].insert(0, 0.)
                # pix_element[0].insert(len(pix_element[0]), 0.)
                pix_element.insert(0, 0.)
                # print("dodano 0")
            # if rest == 1:
            #     pix_element[0].insert(0, 0.)
        # print(" po : \n", pix_element)
    return pix_list



def abcde(pix_list):
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    df = pd.DataFrame.from_records(pix_list)
    # df.add_prefix('col_')
    # df.sort_values(by="word")
    return df

if __name__ == '__main__':
    # creating wav files (./name/name_word_x_iteration1.wav)
    # words = ['hello', 'who', 'which', 'where', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'one', 'two', 'three',
    #          'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'red', 'blue', 'green', 'white', 'black', 'orange',
    #          'pink', 'yellow', 'hours', 'duck', 'pig', 'cow', 'chicken', 'dog', 'cat', 'mouse', 'rat', 'bee', 'bat',
    #          'lion', 'house', 'block', 'flat', 'line', 'square', 'triangle', 'circle', 'cube']
    #
    # files, directory = creat_base_of_files(words, t=2)


    # one file chunks wav, creating img, normalization, conv apply karnels -> new img
    # full_name_file = './' + 'karolski' + '/' + 'karolski_word_yellow_iteration_2.wav'
    # loaded_files = load_sound(full_name_file)
    # chunks_list = get_chunks(loaded_files)
    # chunk_formants = get_formants_trayectory(chunks_list)
    # normalization = get_scaling(chunk_formants)
    # edge_detection = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    # sharpen_image = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    # list_krenels2 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # reduced_image = compare_kernels(np.array(normalization).T, edge_detection, sharpen_image, list_krenels2)
    # new_image = apply_conv(np.array(normalization).T, edge_detection, sharpen_image, list_krenels2)
    # plt.figure().savefig(full_name_file + '.png')


    # ######################### NOW

    pix_list = creating_list_of_pixels()
    fixed_list = same_file_length(pix_list)
    df = abcde(fixed_list)

    x_train = df.iloc[:, 0:33] ##or [:, :-1]
    y_train = df.iloc[:, 33:]

    #Logistic Regression
    log_reg = linear_model.LogisticRegression()
    X_train, x_test, Y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    logic_reg_model = log_reg.fit(X_train, Y_train)
    LogisticRegression_prediction = log_reg.predict(x_test)
    LogisticRegression_score = log_reg.score(x_test, y_test)
    print("Score of Logistic Regression : ", LogisticRegression_score)

    #Decision Tree
    D_tree = tree.DecisionTreeClassifier()
    D_tree_ = D_tree.fit(X_train, Y_train)
    y_pred = D_tree.predict(x_test)
    print("Decision Tree : ")
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    #Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, Y_train)
    rf_pred = rf.predict(x_test)
    print("Random forest :", metrics.accuracy_score(y_test, rf_pred))

    #SVM
    SVM_ = svm.SVC(kernel='linear')
    SVM_.fit(X_train, Y_train)
    svm_pred = SVM_.predict(x_test)
    SVM_.score(x_test, y_test)
    print("SMV score :", metrics.accuracy_score(y_test, y_pred))


    # lm = linear_model.LinearRegression()
    # model = lm.fit(X_train, Y_train)
    # prediction = lm.predict(x_test)
    # plt.scatter(y_test, prediction)
    # plt.xlabel("True Values")
    # plt.ylabel("Predictions")
    # print("Score : ", model.score(x_test, y_test))

    # plt.subplot(2, 1, 1)
    # plt.imshow(reduced_image.T, cmap='gray')
    # plt.subplot(2, 1, 2)
    # plt.imshow(np.array(normalization).T, cmap='gray')
    #
    # plt.title(full_name_file)
    # plt.show()
