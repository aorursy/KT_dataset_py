import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))
import pandas as pd

import librosa

import librosa.display

import glob 

import matplotlib.pyplot as plt
set_a = pd.read_csv("/kaggle/input/heartbeat-sounds/set_a.csv")

set_b = pd.read_csv("/kaggle/input/heartbeat-sounds/set_b.csv")

frames = [set_a, set_b]

train_ab=pd.concat(frames)

train_ab.info()
nb_classes=train_ab.label.unique()

print (nb_classes)
import IPython.display as ipd

normal_file="/kaggle/input/heartbeat-sounds/set_a/normal__201106111136.wav"

ipd.Audio(normal_file) 
from scipy.io import wavfile

rate, data = wavfile.read(normal_file)

print("Sampling (frame) rate = ", rate)

print("Total samples (frames) = ", data.shape)

print(data)
# plot wave by audio frames

plt.figure(figsize=(16, 3))

plt.plot(data, '-', );
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

#print (mfccs)

plt.figure(figsize=(12, 3))

librosa.display.specshow(mfccs, x_axis='time')

plt.colorbar()

plt.title('Mel-frequency cepstral coefficients (MFCCs)')

plt.tight_layout()
print("Number of training examples=", train_ab.shape[0], "  Number of classes=", len(train_ab.label.unique()))
SAMPLE_RATE = 16000

# seconds

MAX_SOUND_CLIP_DURATION=12 



import numpy as np



def audio_norm(data):

    max_data = np.max(data)

    min_data = np.min(data)

    data = (data-min_data)/(max_data-min_data+0.0001)

    return data-0.5



# get audio data without padding highest qualify audio

def load_file_data_without_change(folder,file_names, duration=3, sr=16000):

    input_length=sr*duration

    # function to load files and extract features

    # file_names = glob.glob(os.path.join(folder, '*.wav'))

    data = []

    for file_name in file_names:

        try:

            sound_file=folder+file_name

            print ("load file ",sound_file)

            # use kaiser_fast technique for faster extraction

            X, sr = librosa.load( sound_file, res_type='kaiser_fast') 

            dur = librosa.get_duration(y=X, sr=sr)

            # extract normalized mfcc feature from data

            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0) 

        except Exception as e:

            print("Error encountered while parsing file: ", file)

        feature = np.array(mfccs).reshape([-1,1])

        data.append(feature)

    return data





# get audio data with a fix padding may also chop off some file

def load_file_data (folder,file_names, duration=12, sr=16000):

    input_length=sr*duration

    # function to load files and extract features

    # file_names = glob.glob(os.path.join(folder, '*.wav'))

    data = []

    for file_name in file_names:

        try:

            sound_file=folder+file_name

            print ("load file ",sound_file)

            # use kaiser_fast technique for faster extraction

            X, sr = librosa.load( sound_file, sr=sr, duration=duration,res_type='kaiser_fast') 

            dur = librosa.get_duration(y=X, sr=sr)

            # pad audio file same duration

            if (round(dur) < duration):

                print ("fixing audio lenght :", file_name)

                y = librosa.util.fix_length(X, input_length)                

            #normalized raw audio 

            # y = audio_norm(y)            

            # extract normalized mfcc feature from data

            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)             

        except Exception as e:

            print("Error encountered while parsing file: ", file)        

        feature = np.array(mfccs).reshape([-1,1])

        data.append(feature)

    return data
# simple encoding of categories, limited to 3 types

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



# Map label text to integer

CLASSES = ['artifact','murmur','normal']

# {'artifact': 0, 'murmur': 1, 'normal': 3}

NB_CLASSES=len(CLASSES)



# Map integer value to text labels

label_to_int = {k:v for v,k in enumerate(CLASSES)}

print (label_to_int)

print (" ")

# map integer to label text

int_to_label = {v:k for k,v in label_to_int.items()}

print(int_to_label)
# load dataset-a, keep them separate for testing purpose

import os, fnmatch

INPUT_DIR = "/kaggle/input/heartbeat-sounds"

A_folder=INPUT_DIR+'/set_a/'

# set-a

A_artifact_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'artifact*.wav')

A_artifact_sounds = load_file_data(folder=A_folder,file_names=A_artifact_files, duration=MAX_SOUND_CLIP_DURATION)

A_artifact_labels = [0 for items in A_artifact_files]



A_normal_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'normal*.wav')

A_normal_sounds = load_file_data(folder=A_folder,file_names=A_normal_files, duration=MAX_SOUND_CLIP_DURATION)

A_normal_labels = [2 for items in A_normal_sounds]



A_extrahls_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'extrahls*.wav')

A_extrahls_sounds = load_file_data(folder=A_folder,file_names=A_extrahls_files, duration=MAX_SOUND_CLIP_DURATION)

A_extrahls_labels = [1 for items in A_extrahls_sounds]



A_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'murmur*.wav')

A_murmur_sounds = load_file_data(folder=A_folder,file_names=A_murmur_files, duration=MAX_SOUND_CLIP_DURATION)

A_murmur_labels = [1 for items in A_murmur_files]



# test files

A_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'Aunlabelledtest*.wav')

A_unlabelledtest_sounds = load_file_data(folder=A_folder,file_names=A_unlabelledtest_files, duration=MAX_SOUND_CLIP_DURATION)

A_unlabelledtest_labels = [-1 for items in A_unlabelledtest_sounds]

# load dataset-b, keep them separate for testing purpose 

B_folder=INPUT_DIR+'/set_b/'

# set-b

B_normal_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'normal*.wav')  # include noisy files

B_normal_sounds = load_file_data(folder=B_folder,file_names=B_normal_files, duration=MAX_SOUND_CLIP_DURATION)

B_normal_labels = [2 for items in B_normal_sounds]



B_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'murmur*.wav')  # include noisy files

B_murmur_sounds = load_file_data(folder=B_folder,file_names=B_murmur_files, duration=MAX_SOUND_CLIP_DURATION)

B_murmur_labels = [1 for items in B_murmur_files]



B_extrastole_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'extrastole*.wav')

B_extrastole_sounds = load_file_data(folder=B_folder,file_names=B_extrastole_files, duration=MAX_SOUND_CLIP_DURATION)

B_extrastole_labels = [1 for items in B_extrastole_files]



#test files

B_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'Bunlabelledtest*.wav')

B_unlabelledtest_sounds = load_file_data(folder=B_folder,file_names=B_unlabelledtest_files, duration=MAX_SOUND_CLIP_DURATION)

B_unlabelledtest_labels = [-1 for items in B_unlabelledtest_sounds]

x_data = np.concatenate((A_artifact_sounds, A_normal_sounds,A_extrahls_sounds,A_murmur_sounds, 

                         B_normal_sounds,B_murmur_sounds,B_extrastole_sounds))



y_data = np.concatenate((A_artifact_labels, A_normal_labels,A_extrahls_labels,A_murmur_labels,

                         B_normal_labels,B_murmur_labels,B_extrastole_labels))



test_x = np.concatenate((A_unlabelledtest_sounds,B_unlabelledtest_sounds))

test_y = np.concatenate((A_unlabelledtest_labels,B_unlabelledtest_labels))



print ("combined training data record: ",len(y_data), len(test_y))
import keras

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, shuffle=True)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, shuffle=True)



# One-Hot encoding for classes

y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))

y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))

y_val = np.array(keras.utils.to_categorical(y_val, len(CLASSES)))

test_y=np.array(keras.utils.to_categorical(test_y, len(CLASSES)))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, LSTM

from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger

from keras.utils import np_utils

from sklearn import metrics 

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing import LabelEncoder

import itertools
model = Sequential()

model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.20, return_sequences=True,input_shape = (40,1)))

model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.20, return_sequences=False))

model.add(Dense(len(CLASSES), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['acc','mse', 'mae', 'mape', 'cosine'])

model.summary()


MAX_PATIENT=12

MAX_EPOCHS=100

MAX_BATCH=32





callback=[ReduceLROnPlateau(patience=MAX_PATIENT, verbose=1)]



history=model.fit(x_train, y_train, 

                  batch_size=MAX_BATCH, 

                  epochs=MAX_EPOCHS,

                  verbose=0,

                  validation_data=(x_val, y_val),

                  callbacks=callback) 

#Plot Keras History

#Plot loss and accuracy for the training and validation set.

def plot_history(history):

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]

    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]

    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:

        print('Loss is missing in history')

        return 

    plt.figure(figsize=(22,10))

    ## As loss always exists

    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Accuracy

    plt.figure(221, figsize=(20,10))

    ## Accuracy

    # plt.figure(2,figsize=(14,5))

    plt.subplot(221, title='Accuracy')

    for l in acc_list:

        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    for l in val_acc_list:    

        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    ## Loss

    plt.subplot(222, title='Loss')

    for l in loss_list:

        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    for l in val_loss_list:

        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))    

    plt.title('Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()



# plot history

plot_history(history)