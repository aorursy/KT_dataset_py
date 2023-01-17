import numpy as np #mathematiclal 

import librosa # Used for signal processing

import librosa.display

import os #library for reading items in a folder

from scipy.io import wavfile # For reading wav files
signals = []

labels = []



male_dir = "../input/zip/Male/"

female_dir = "../input/zip/Female/"



# get items in the male and female directory

male_items = os.listdir(male_dir)

female_items = os.listdir(female_dir)



for item in male_items:

    if(item.find(".wav") != -1):

        fs, data = wavfile.read(male_dir + item)

        signals.append(data)

        labels.append(0)



for item in female_items:

    if(item.find(".wav") != -1):

        fs, data = wavfile.read(female_dir + item)

        signals.append(data)

        labels.append(1)
print(len(male_items), len(female_items))
float_signals = []

for signal in signals:

    newsignal = []

    for point in signal:

        newsignal.append(float(point))

    float_signals.append(newsignal)
float_signals = np.array(float_signals)
import matplotlib.pyplot as plt # plotting library

ffts = []

mfccs = []

mfccs_flatten = []

mels = []

mels_flatten = []

for i in range(len(signals)):

    mfccs.append(librosa.feature.mfcc(y=float_signals[i]))

    #ffts.append(np.log(abs(np.fft.fft(signals[i]))))

    mfccs_flatten.append(np.asarray(mfccs[i]).flatten())

    #mels.append(librosa.feature.melspectrogram(y=float_signals[i]))

    #mels_flatten.append(np.asarray(mels[i]).flatten())

    
#S_dB = librosa.power_to_db(mels[2], ref=np.max)

#librosa.display.specshow(S_dB, x_axis='time', y_axis='mel',fmax=8000)

#librosa.display.specshow(mfccs[0], x_axis='time')

print(names[labels[2]])
#S_dB = librosa.power_to_db(mels[41], ref=np.max)

#librosa.display.specshow(S_dB, x_axis='time', y_axis='mel',fmax=8000)

librosa.display.specshow(mfccs[40], x_axis='time')

print(names[labels[41]])
from sklearn.model_selection import train_test_split #function that allows us to split data for training and testing

X_train, X_test, y_train, y_test = train_test_split(mfccs_flatten, labels, train_size=.8, shuffle=True)

print("Training samples", len(X_train))

print("Test samples", len(X_test))
print(len([x for x in y_train if x == 1]))

print(len([x for x in y_train if x == 0]))

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import confusion_matrix 

y_pred = neigh.predict(X_test)

print(accuracy_score(y_pred,y_test))

print(balanced_accuracy_score(y_pred, y_test))

print(confusion_matrix(y_test, y_pred))