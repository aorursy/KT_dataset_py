# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! pip install python_speech_features
from python_speech_features import mfcc

from python_speech_features import logfbank

import scipy.io.wavfile as wav



(rate,sig) = wav.read("/kaggle/input/shifts130_iso_hdpump_3.wav")

mfcc_feat = mfcc(sig,rate)

fbank_feat = logfbank(sig,rate)



print(fbank_feat[1:3,:])
#python_speech_features.base.mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=<function <lambda>>)


#def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True):

 #   pass

   
os.listdir('../input/')
f = '/kaggle/input/shifts130_iso_hdpump_3.wav'

f.split('/')[-1]
import librosa

audio_path = '/kaggle/input/shifts130_iso_hdpump_3.wav'

x , sr = librosa.load(audio_path)

print(type(x), type(sr))
librosa.load(audio_path, sr=44100)

# librosa.load(audio_path, sr=none)

import IPython.display as ipd

ipd.Audio(audio_path)
#display waveform

%matplotlib inline

import matplotlib.pyplot as plt

import librosa.display

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x, sr=sr)
#display Spectrogram

X = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 

#If to pring log of frequencies  

#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

plt.colorbar()
x, sr = librosa.load(audio_path)

#Plot the signal:

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x, sr=sr)
# Zooming in

n0 = 9000

n1 = 9100

plt.figure(figsize=(14, 5))

plt.plot(x[n0:n1])

plt.grid()
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)

print(sum(zero_crossings))
#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound

import sklearn

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

spectral_centroids.shape

# Computing the time variable for visualization

frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_centroids), color='r')
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_rolloff), color='r')
mfccs = librosa.feature.mfcc(x, sr=sr)

print(mfccs.shape)

#Displaying  the MFCCs:

librosa.display.specshow(mfccs, sr=sr, x_axis='time')
mfccs
mfccs.shape
header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth spectral_rolloff zero_crossing_rate'



for i in range(1, 21):

    header += f' mfcc{i}'

header += ' label'

header = header.split()
!ls
import csv
file = open('data.csv', 'w', newline='')

with file:

    writer = csv.writer(file)

    writer.writerow(header)



genres = 'fail new shifts130 shift50'.split()

for g in genres:

    for filename in os.listdir(f'../input/'):

        songname = '../input/' + filename

        y, sr = librosa.load(songname, mono=True, duration=30)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        rms = librosa.feature.rms(y=y)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        zcr = librosa.feature.zero_crossing_rate(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        to_append = f'{filename} {np.mean (chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    

        for e in mfcc:

            to_append += f' {np.mean(e)}'

        to_append += f' {g}'

        file = open('data.csv', 'a', newline='')

        with file:

            writer = csv.writer(file)

            writer.writerow(to_append.split())
data = pd.read_csv('data.csv', error_bad_lines=False)

data