# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
%matplotlib inline
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from collections import defaultdict, Counter
from scipy import signal
import numpy as np
import librosa
import random as rn
import tensorflow as tf
from tensorflow import keras

DATA_DIR = '/kaggle/input/spoken-digit-dataset/free-spoken-digit-dataset-master/free-spoken-digit-dataset-master/recordings/'
# random_file = rn.choice(os.listdir(DATA_DIR))
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
wav, sr = librosa.load('/kaggle/input/spoken-digit-dataset/free-spoken-digit-dataset-master/free-spoken-digit-dataset-master/recordings/0_jackson_14.wav')
print(sr,wav.shape)
plt.plot(wav)
librosa.get_duration(y=wav,sr=sr)
11265/22050
import librosa.display
D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

spectrogram = np.abs(librosa.stft(wav))
padded_spectogram = pad2d(spectrogram,40)

D = librosa.amplitude_to_db(padded_spectogram, ref=np.max)
librosa.display.specshow(D, y_axis='linear')
padded_spectogram = pad2d(spectrogram,60)

D = librosa.amplitude_to_db(padded_spectogram, ref=np.max)
librosa.display.specshow(D, y_axis='linear')
mel_spectrogram = librosa.feature.melspectrogram(wav)


D = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
librosa.display.specshow(D, y_axis='linear')
padded_mel_spectrogram = pad2d(mel_spectrogram,40)
D = librosa.amplitude_to_db(padded_mel_spectrogram, ref=np.max)
librosa.display.specshow(D)
mfcc = librosa.feature.mfcc(wav)
padded_mfcc = pad2d(mfcc,40)
D = librosa.amplitude_to_db(mfcc, ref=np.max)
librosa.display.specshow(D)
D = librosa.amplitude_to_db(padded_mfcc, ref=np.max)
librosa.display.specshow(D)
from keras.utils import to_categorical
import os

test_speaker = 'theo'
train_X = []
train_spectrograms = []
train_mel_spectrograms = []
train_mfccs = []
train_y = []

test_X = []
test_spectrograms = []
test_mel_spectrograms = []
test_mfccs = []
test_y = []
train_mfccs_no_padding=[]
test_mfccs_no_padding=[]


for fname in os.listdir(DATA_DIR):
    try:
        if '.wav' not in fname or 'dima' in fname:
            continue
        struct = fname.split('_')
        digit = struct[0]
        speaker = struct[1]
        wav, sr = librosa.load(DATA_DIR + fname)
#         wav, sr = librosa.load(DATA_DIR + fname, mono=True, sr=8000, duration = 1.024)
#         wav, sr = librosa.load(DATA_DIR + fname, mono=True, sr=8000)
        wav = librosa.util.normalize(wav) #normalize 
    
        padded_x = pad1d(wav, 30000)
        spectrogram = np.abs(librosa.stft(wav))
        padded_spectogram = pad2d(spectrogram,40)

        mel_spectrogram = librosa.feature.melspectrogram(wav)
        padded_mel_spectrogram = pad2d(mel_spectrogram,40)

        mfcc = librosa.feature.mfcc(wav)
#         duration = wav.shape[0]/sr
#         speed_change = 2.0* duration/1.024
#         print('duration',duration)
#         print('speed_change',speed_change)
#         wav = librosa.effects.time_stretch(wav, speed_change)
#         wav = wav[:4096]
#         mfcc = librosa.feature.mfcc(wav, sr=sr, n_mfcc=40, hop_length=int(0.048*sr), n_fft=int(0.096*sr))
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

#         print("shape=",mfcc.shape[1], wav.shape[0])
        max_pad_len=11
#         pad_width = max_pad_len - mfcc.shape[1]
        padded_mfcc = pad2d(mfcc,40)
#         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        if speaker == test_speaker:
            test_X.append(padded_x)
            test_spectrograms.append(padded_spectogram)
            test_mel_spectrograms.append(padded_mel_spectrogram)
            test_mfccs.append(padded_mfcc)
            test_y.append(digit)
            test_mfccs_no_padding.append(mfcc)
        else:
            train_X.append(padded_x)
            train_spectrograms.append(padded_spectogram)
            train_mel_spectrograms.append(padded_mel_spectrogram)
            train_mfccs.append(padded_mfcc)
            train_mfccs_no_padding.append(mfcc)
            train_y.append(digit)
    except Exception as e:
        print(fname, e)
        raise

train_X = np.vstack(train_X)
train_spectrograms = np.array(train_spectrograms)
train_mel_spectrograms = np.array(train_mel_spectrograms)
train_mfccs = np.array(train_mfccs)
# train_mfccs_no_padding=np.array(train_mfccs_no_padding)
train_y = to_categorical(np.array(train_y))

test_X = np.vstack(test_X)
test_spectrograms = np.array(test_spectrograms)
test_mel_spectrograms = np.array(test_mel_spectrograms)
test_mfccs = np.array(test_mfccs)
# test_mfccs_no_padding=np.array(test_mfccs_no_padding)
test_y = to_categorical(np.array(test_y))


print('train_X:', train_X.shape)
print('train_y:', train_y.shape)
print('test_X:', test_X.shape)
print('test_y:', test_y.shape)
np.shape(train_mfccs)
len(train_mfccs_no_padding)
train_mfccs.shape
train_X_ex = np.expand_dims(train_mfccs, -1)
test_X_ex = np.expand_dims(test_mfccs, -1)
print('train X shape:', train_X_ex.shape)
print('test X shape:', test_X_ex.shape)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.002), 
                         input_shape=(20, 40, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(64, (1,1), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Conv2D(64, (1,1), activation=tf.keras.layers.LeakyReLU()),
#   tf.keras.layers.MaxPooling2D(2,2),  
#   tf.keras.layers.Dropout(0.1),  
    
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.002)),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',metrics=['accuracy'])

# model.fit(train_X, train_y, epochs=5,verbose =1)
history = model.fit(train_X_ex,
          train_y,
          epochs=20,
          batch_size=64,
          validation_data=(test_X_ex, test_y))
predictions = model.predict(test_X_ex)
results=np.argmax(predictions,axis=1)
test_X_ex.shape
# results
from sklearn.metrics import classification_report
print(classification_report(test_y, to_categorical(results)))
print(train_mfccs.shape)
print(test_mfccs.shape)
train_mfccs.shape[1],train_mfccs.shape[2]

input_shape=(20, 40)   

lstm_model = tf.keras.models.Sequential([
tf.keras.layers.LSTM(64, input_shape=(20, 40), return_sequences=True, 
                     kernel_initializer=tf.keras.initializers.he_uniform(),
#                      kernel_regularizer=tf.keras.regularizers.l2(0.002)
                    ),
   
tf.keras.layers.LSTM(32,return_sequences=False, 
                      kernel_initializer=tf.keras.initializers.he_uniform(),
                      kernel_regularizer=tf.keras.regularizers.l2(0.01)
                    ),
# tf.keras.layers.BatchNormalization() ,    
tf.keras.layers.Dropout(0.5),   
tf.keras.layers.Dense(32, activation='relu', 
                       kernel_initializer=tf.keras.initializers.he_uniform(),
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)
                     ),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(10, activation='softmax')
])

lstm_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.008),
          loss = 'categorical_crossentropy',metrics=['accuracy'])

# model.fit(train_X, train_y, epochs=5,verbose =1)
history = lstm_model.fit(train_mfccs,
      train_y,
      epochs=30,
      batch_size=32,
      validation_data=(test_mfccs, test_y))
predictions = lstm_model.predict(test_mfccs)
results=np.argmax(predictions,axis=1)
from sklearn.metrics import classification_report
print(classification_report(test_y, to_categorical(results)))
