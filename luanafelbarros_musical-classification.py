# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import librosa

import librosa.display





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def import_signal(path):

    s, sr = librosa.core.load(path)

    return s



def plot_signals(s, index):

    fig, a = plt.subplots(1, figsize = (10, 8))

    title = "sinal do áudio {}".format(index+1)

    plt.xlabel('Time')

    plt.ylabel('Amplitude')

    plt.title(title)

    a.plot(s)





paths = ['/kaggle/input/audios/audio1.wav', '/kaggle/input/audios/audio2.wav']

signals = []

for p in paths:

    signals.append(import_signal(p))

signals = np.asarray(signals)

print(signals.shape)



for index, s in enumerate(signals):

    plot_signals(s, index)

    



# return magnitude S

def stft(signal):

    S, phase = librosa.magphase(np.abs(librosa.stft(signal, hop_length=1024)))

    return S



def plot_spect(index, S):

    plt.figure(figsize=(10, 8))

    librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis='log', x_axis='time')

    title = "spectrogram audio {}".format(index+1)

    plt.title(title)

    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()

    plt.show()



signals_stft = []

for s in signals:

    signals_stft.append(stft(s))

    

signals_stft = np.asarray(signals_stft)

signals_stft.shape



for index, s in enumerate(signals_stft):

    plot_spect(index, s)
def get_centroid(S):

    return librosa.feature.spectral_centroid(S=S)



def plot_centroids(c, index):

    fig, a = plt.subplots(1, figsize = (10, 8))

    title = "centroides audio {}".format(index+1)

    plt.title(title)

    a.plot(c)



centroids = []

for s in signals_stft:

    c = get_centroid(s)

    centroids.append(c[0])

centroids = np.asarray(centroids)

    

for index, c in enumerate(centroids):

    plot_centroids(c, index)

    

def get_flatness(S):

    return librosa.feature.spectral_flatness(S=S)



def plot_flatness(f, index):

    fig, a = plt.subplots(1, figsize = (10, 8))

    title = "flatness audio {}".format(index+1)

    plt.title(title)

    a.axis([0, 600, 0, 0.125])

    a.plot(f)

    



flatness = []

for s in signals_stft:

    f = get_flatness(s)

    flatness.append(f[0])

flatness = np.asarray(flatness)

    

for i, f in enumerate(flatness):

    plot_flatness(f, i)

    

def get_rms(s):

    return librosa.feature.rms(s, hop_length=1024)



def plot_rms(x, y):

    print(y)

    fig, a = plt.subplots(1, figsize = (10, 8))

    a.axis([0, 4, 0, 2])

    a.plot(x, y, 'ro')

    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    plt.xlabel('Áudio')

    plt.ylabel('RMS')

    plt.title('Valor RMS para os sinais de áudio')

    for a,b in zip(x, y): 

        plt.text(a, b, '    ' + str(b), horizontalalignment='left', verticalalignment='bottom')

    

    

rms_arr = []

indexes = []

for i, s in enumerate(signals_stft):

    rms = get_rms(s)

    rms_arr.append(round(rms[0][0],3))

    indexes.append(i+1)

plot_rms(indexes, rms_arr)
info_signals = []

for c, f, r in zip(centroids, flatness, rms_arr):

    info_signals.append([np.mean(c), np.std(c), np.mean(f), np.std(f), r])



print(info_signals)



rotulos = np.array([0,1])

    
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(info_signals, rotulos)

y = knn.predict(info_signals)

print(y)