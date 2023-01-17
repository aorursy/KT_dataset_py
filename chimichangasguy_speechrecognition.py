# Use "conda install -c conda-forge python-sounddevice" for installing sounddevice for recording
import os

import librosa   #for audio processing

import IPython.display as ipd

import matplotlib.pyplot as plt

import numpy as np

from scipy.io import wavfile #for audio processing

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D

#We use 1 dimensional arrays due to audio files, we use Conv1D and MaxPooling1D

#Dense is used for connecting the entire neural network

#Flattening is the process of converting all the resultant 1D arrays into a single long continuous linear vector

#A dropout layer is used for regularization where you randomly set some of the dimensions of your input vector to be zero

from keras.models import Model #Used to instantiate a Model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from matplotlib import pyplot 

import random

import soundfile as sf
train_audio_path = '../input/tensorflow-speech-recognition-challenge/train/audio/'

samples, sample_rate = librosa.load(train_audio_path+'yes/0a7c2a8d_nohash_0.wav', sr = 16000)

fig = plt.figure(figsize=(14, 8)) #FigSize determines the pixels of the plot

ax1 = fig.add_subplot(211) #Subplot grid parameter of 2x1 at the top left of the plot

ax1.set_title('Raw wave of ' + '../input/train/audio/yes/0a7c2a8d_nohash_0.wav')

ax1.set_xlabel('Time')

ax1.set_ylabel('Amplitude')

ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
ipd.Audio(samples, rate=sample_rate)

print(sample_rate)
samples = librosa.resample(samples, sample_rate, 8000)

ipd.Audio(samples, rate=8000)
labels=os.listdir(train_audio_path)



#find count of each label and plot bar graph

no_of_recordings=[]

for label in labels:

    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]

    no_of_recordings.append(len(waves))

    

#plot

plt.figure(figsize=(30,5))

index = np.arange(len(labels))

plt.bar(index, no_of_recordings)

plt.xlabel('Commands', fontsize=12)

plt.ylabel('No of recordings', fontsize=12)

plt.xticks(index, labels, fontsize=15, rotation=60)

plt.title('No. of recordings for each command')

plt.show()



labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "wow", "house", "happy", "tree", "dog", "cat", "bird", 

        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
duration_of_recordings=[]

for label in labels:

    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]

    for wav in waves:

        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)

        duration_of_recordings.append(float(len(samples)/sample_rate))

    

plt.hist(np.array(duration_of_recordings))
train_audio_path = '../input/tensorflow-speech-recognition-challenge/train/audio/'



all_wave = []

all_label = []

for label in labels:

    print(label)

    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]

    for wav in waves:

        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)

        samples = librosa.resample(samples, sample_rate, 8000)

        if(len(samples)== 8000) : 

            all_wave.append(samples)

            all_label.append(label)
le = LabelEncoder()

y=le.fit_transform(all_label)

classes= list(le.classes_)
from keras.utils import np_utils

y=np_utils.to_categorical(y, num_classes=len(labels))

all_wave = np.array(all_wave).reshape(-1,8000,1)
from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
K.clear_session()



inputs = Input(shape=(8000,1))



#First Conv1D layer

conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)

conv = MaxPooling1D(3)(conv)

conv = Dropout(0.3)(conv)



#Second Conv1D layer

conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)

conv = MaxPooling1D(3)(conv)

conv = Dropout(0.3)(conv)



#Third Conv1D layer

conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)

conv = MaxPooling1D(3)(conv)

conv = Dropout(0.3)(conv)



#Fourth Conv1D layer

conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)

conv = MaxPooling1D(3)(conv)

conv = Dropout(0.3)(conv)



#Flatten layer

conv = Flatten()(conv)



#Dense Layer 1

conv = Dense(256, activation='relu')(conv)

conv = Dropout(0.3)(conv)



#Dense Layer 2

conv = Dense(128, activation='relu')(conv)

conv = Dropout(0.3)(conv)



outputs = Dense(len(labels), activation='softmax')(conv)



model = Model(inputs, outputs)

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 

mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))
pyplot.plot(history.history['loss'], label='train') 

pyplot.plot(history.history['val_loss'], label='test') 

pyplot.legend()

pyplot.show()
'''from keras.models import load_model

model=load_model('best_model.hdf5')'''
def predict(audio):

    prob=model.predict(audio.reshape(1,8000,1))

    index=np.argmax(prob[0])

    return classes[index]
index=random.randint(0,len(x_val)-1)

samples=x_val[index].ravel()

print("Audio:",classes[np.argmax(y_val[index])])

ipd.Audio(samples, rate=8000)

print("Text:",predict(samples))
os.listdir('../input/tensorflow-speech-recognition-challenge/train/audio/right')

filepath='../input/tensorflow-speech-recognition-challenge/train/audio/right'
#import wave

#reading the voice commands

#Audio should not be of 24 sample size (8,16 or 32) use AUDACITY to record or edit

#samples = wave.open(filepath + '/' + 'stop.wav', 'wb')

#samples.setnchannels(1) # mono

samples, sample_rate = librosa.load(filepath + '/' + '4c7c95de_nohash_1.wav', sr = 16000)

samples = librosa.resample(samples, sample_rate, 8000)

ipd.Audio(samples,rate=8000)  
predict(samples)