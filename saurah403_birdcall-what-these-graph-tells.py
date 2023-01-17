import os



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import librosa

import librosa.display

import IPython.display as ipd



import sklearn



import warnings

warnings.filterwarnings('ignore')
train_audio_dir = '../input/birdsong-recognition/train_audio'

train = pd.read_csv('../input/birdsong-recognition/train.csv')
base_dir = '../input/birdsong-recognition/train_audio/'

train['full_path'] = base_dir + train['ebird_code'] + '/'+ train['filename']
train[train['ebird_code']== 'amered'].sample(1, random_state = 33)['full_path'].values[0]
amered = train[train['ebird_code']== 'amered'].sample(1, random_state = 33)['full_path'].values[0]

pingro = train[train['ebird_code'] == "pingro"].sample(1, random_state = 33)['full_path'].values[0]

vesspa = train[train['ebird_code'] == "vesspa"].sample(1, random_state = 33)['full_path'].values[0]





audio_file, _ = librosa.effects.trim(y)



# the result is an numpy ndarray

print('Audio File:', audio_file, '\n')

print('Audio File shape:', np.shape(audio_file))
ipd.Audio(amered)
audio_amered, sr = librosa.load(amered)
n_fft = 2048 # FFT window size

hop_length = 512



D_amered = np.abs(librosa.stft(audio_amered, n_fft = n_fft, hop_length = hop_length))

DB_amered = librosa.amplitude_to_db(D_amered, ref = np.max)

librosa.display.specshow(DB_amered, sr = sr, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool')


plt.Figure(figsize=(16,9))

plt.title(('Sound waves'), fontsize=16)



librosa.display.waveplot(y= audio_amered, sr = sr, color = "#A300F9")
n0 = 9000

n1 = 9100

plt.figure(figsize=(14, 5))

plt.plot(audio_amered[n0:n1])

plt.grid()
zero_amered = librosa.zero_crossings(audio_amered, pad=False)

print('change rate {}'.format(sum(zero_amered)))
spectral_centroids = librosa.feature.spectral_centroid(audio_amered, sr=sr)[0]



# Shape is a vector

print('Centroids:', spectral_centroids, '\n')

print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')



# Computing the time variable for visualization

frames = range(len(spectral_centroids))



# Converts frame counts to time (seconds)

t = librosa.frames_to_time(frames)



print('frames:', frames, '\n')

print('t:', t)



# Function that normalizes the Sound Data

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)
plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_amered, sr=sr, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(spectral_centroids), color='#FFB100', lw=2)

plt.legend(["Spectral Centroid", "Wave"])

plt.title("Spectral Centroid: Cangoo Bird", fontsize=16);
spectral_rolloff = librosa.feature.spectral_rolloff(audio_amered, sr=sr)[0]





frames = range(len(spectral_rolloff))



t = librosa.frames_to_time(frames)





plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_amered, sr=sr, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(spectral_rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: Amered Bird", fontsize=16);