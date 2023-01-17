# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import librosa

from librosa import feature

import librosa.display



import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import keras

from keras.layers import *

from keras import metrics

from keras.utils import to_categorical



import sklearn

from sklearn.model_selection import train_test_split



import csv



import IPython.display as ipd



firstfilepath = '../input/tabla-taala-dataset/tablaDataset/addhatrital/addhatrital01.wav'

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x, sr = librosa.load(firstfilepath)

print(type(x), type(sr))
ipd.Audio(firstfilepath)
plt.figure(figsize=(14, 5))

librosa.display.waveplot(x, sr=sr)
chromagram = librosa.feature.chroma_stft(x, sr=sr)

plt.figure(figsize=(15, 5))

librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma')
example_rmse = librosa.feature.rms(x)

print(example_rmse.shape)

print(np.mean(example_rmse))
example_spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

example_spectral_centroids.shape

frames = range(len(example_spectral_centroids))

t = librosa.frames_to_time(frames)

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(example_spectral_centroids), color='r')

print(np.mean(example_spectral_centroids))
example_spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr=sr)[0]

example_spectral_bandwidth.shape

frames = range(len(example_spectral_bandwidth))

t = librosa.frames_to_time(frames)

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(example_spectral_bandwidth), color='r')

print(np.mean(example_spectral_bandwidth))
example_spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]

example_spectral_rolloff.shape

frames = range(len(example_spectral_rolloff))

t = librosa.frames_to_time(frames)

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(example_spectral_bandwidth), color='r')

print(np.mean(example_spectral_rolloff))
n0 = 9000

n1 = 9100

plt.figure(figsize=(14, 5))

plt.plot(x[n0:n1])

plt.grid()

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)

print(sum(zero_crossings))
mfccs = librosa.feature.mfcc(x, sr=sr)

print(mfccs.shape)

librosa.display.specshow(mfccs, sr=sr, x_axis='time')

print(np.mean(mfccs))
header = 'chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

for i in range(1, 21):

    header += f' mfcc{i}'

header += ' label'

header = header.split()
file = open('dataset.csv', 'w', newline='')

with file:

    writer = csv.writer(file)

    writer.writerow(header)

taals = 'addhatrital bhajani dadra deepchandi ektal jhaptal rupak trital'.split()

for t in taals:

    for filename in os.listdir(f'../input/tabla-taala-dataset/tablaDataset/{t}'):

        taalfile = f'../input/tabla-taala-dataset/tablaDataset/{t}/{filename}'

        y, sr = librosa.load(taalfile, mono=True, duration=30)

        rms = librosa.feature.rms(y=y)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        zcr = librosa.feature.zero_crossing_rate(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        to_append = f' {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '    

        for e in mfcc:

            to_append += f' {np.mean(e)}'

        to_append += f' {t}'

        file = open('dataset.csv', 'a', newline='')

        with file:

            writer = csv.writer(file)

            writer.writerow(to_append.split())
data = pd.read_csv('dataset.csv')

print(data)
d = dict(zip(taals, range(0,8)))

d
data['label'] = data['label'].map(d)

print(data)
y = data['label']

X = data.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

print(y_train)

print(y_test)


model = keras.models.Sequential()

model.add(Dense(128, input_dim = 26, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 10, epochs = 300, verbose = 1)
loss = model.evaluate(X_test, y_test, verbose=1)
model.summary()