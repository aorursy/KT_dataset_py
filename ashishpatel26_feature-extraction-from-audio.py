import warnings

warnings.filterwarnings('ignore')
import librosa

audio_path = '../input/train_curated/0006ae4e.wav'

x , sr = librosa.load(audio_path)
import IPython.display as ipd

ipd.Audio(audio_path)
%matplotlib inline

import sklearn

import matplotlib.pyplot as plt

import librosa.display



plt.figure(figsize=(20, 5))

librosa.display.waveplot(x, sr=sr)
X = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(20, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

plt.colorbar()
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

plt.colorbar()
import numpy as np

sr = 22050 # sample rate

T = 5.0    # seconds

t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable

x = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 Hz

ipd.Audio(x, rate=sr) # load a NumPy array
librosa.output.write_wav('../tone_440.wav', x, sr) # writing wave file in tone440.wav format
x, sr = librosa.load('../input/train_noisy/000b6cfb.wav')

ipd.Audio(x, rate=sr)
#Plot the signal:

plt.figure(figsize=(20, 5))

librosa.display.waveplot(x, sr=sr)
# Zooming in

n0 = 9000

n1 = 9100

plt.figure(figsize=(20, 5))

plt.plot(x[n0:n1])

plt.grid()
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)

zero_crossings.shape
print(sum(zero_crossings))
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

spectral_centroids.shape
# Computing the time variable for visualization

plt.figure(figsize=(20,5))

frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames)



# Normalising the spectral centroid for visualisation

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)



#Plotting the Spectral Centroid along the waveform

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_centroids), color='r')
plt.figure(figsize=(20,5))

spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_rolloff), color='r')

plt.grid()
plt.figure(figsize=(20,5))

x, fs = librosa.load('../input/train_curated/0006ae4e.wav')

librosa.display.waveplot(x, sr=sr)
# MFCC

plt.figure(figsize=(20,5))

mfccs = librosa.feature.mfcc(x, sr=sr)

print(mfccs.shape)



librosa.display.specshow(mfccs, sr=sr, x_axis='time')
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

print(mfccs.mean(axis=1))

print(mfccs.var(axis=1))
plt.figure(figsize=(20,8))

librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# Loadign the file

x, sr = librosa.load('../input/train_curated/0006ae4e.wav')

ipd.Audio(x, rate=sr)
hop_length = 512

chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)

plt.figure(figsize=(15, 5))

librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')