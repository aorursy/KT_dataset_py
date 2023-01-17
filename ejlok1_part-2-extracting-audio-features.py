import pandas as pd       

import os 

import math 

import numpy as np

import matplotlib.pyplot as plt  

import IPython.display as ipd  # To play sound in the notebook

import librosa

import librosa.display

!apt install -y ffmpeg

# os.chdir("/kaggle/input/freesound-audio-tagging/audio_train")

#os.getcwd()

os.chdir("/kaggle/input/speech-accent-archive/recordings")

# print(os.listdir("/kaggle/input/freesound-audio-tagging/audio_train/audio_train/"))

# print(os.listdir("recordings"))
# Play female from Kentucky

fname_f = 'recordings/' + 'english385.mp3'   

ipd.Audio(fname_f)
# Play male from Kentucky

fname_m = 'recordings/' + 'english381.mp3'

ipd.Audio(fname_m)
# The full 'high fidelity' sampling rate of 44k 

SAMPLE_RATE = 44100

fname_f = 'recordings/' + 'english385.mp3' 

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)



plt.figure(figsize=(12, 3))

plt.figure()

librosa.display.waveplot(y, sr=sr)

plt.title('Audio sampled at 44100 hrz')
# The very 'low fidelity' sampling rate of 6k 

SAMPLE_RATE = 6000

fname_f = 'recordings/' + 'english385.mp3' 

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)



plt.figure(figsize=(12, 3))

plt.figure()

librosa.display.waveplot(y, sr=sr)

plt.title('Audio sampled at 6000 hrz')
# The very very 'low fidelity' sampling rate of 1k 

SAMPLE_RATE = 1000

fname_f = 'recordings/' + 'english385.mp3' 

y, sr = librosa.core.load(fname_f, sr=SAMPLE_RATE, duration = 5)



plt.figure(figsize=(12, 3))

plt.figure()

librosa.display.waveplot(y, sr=sr)

plt.title('Audio sampled at 1000 hrz')
# The 'low fidelity' sampling rate of 6k 

SAMPLE_RATE = 6000

fname_m = 'recordings/' + 'english381.mp3' 

y, sr = librosa.load(fname_m, sr=SAMPLE_RATE, duration = 5)



plt.figure(figsize=(12, 3))

plt.figure()

librosa.display.waveplot(y, sr=sr)

plt.title('Audio sampled at 6000 hrz')
# MFCC for female 

SAMPLE_RATE = 22050

fname_f = 'recordings/' + 'english385.mp3'  

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5) # Chop audio at 5 secs... 

mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc = 5) # 5 MFCC components



plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc)

plt.ylabel('MFCC')

plt.colorbar()
# MFCC for male  

SAMPLE_RATE = 22050

fname_m = 'recordings/' + 'english381.mp3'  

y, sr = librosa.load(fname_m, sr=SAMPLE_RATE, duration = 5)

mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc = 5)



plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc)

plt.ylabel('MFCC')

plt.colorbar()
# Log Mel-spectogram for female 

SAMPLE_RATE = 22050

fname_f = 'recordings/' + 'english385.mp3'  

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5) # Chop audio at 5 secs... 

melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)



# Convert to log scale (dB). We'll use the peak power (max) as reference.

log_S = librosa.amplitude_to_db(melspec)



# Display the log mel spectrogram

plt.figure(figsize=(12,4))

librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

plt.title('Log mel spectrogram for female')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()
# Log Mel-spectogram for male 

SAMPLE_RATE = 22050

fname_m = 'recordings/' + 'english381.mp3'  

y, sr = librosa.load(fname_m, sr=SAMPLE_RATE, duration = 5) # Chop audio at 5 secs... 

melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)



# Convert to log scale (dB). We'll use the peak power (max) as reference.

log_S = librosa.amplitude_to_db(melspec)



# Display the log mel spectrogram

plt.figure(figsize=(12,4))

librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

plt.title('Log mel spectrogram for male')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()
SAMPLE_RATE = 22050

fname_f = 'recordings/' + 'english385.mp3'  

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5) 

y_harmonic, y_percussive = librosa.effects.hpss(y)



ipd.Audio(y_harmonic, rate=sr)
ipd.Audio(y_percussive, rate=sr)
# harmonic 

melspec = librosa.feature.melspectrogram(y_harmonic, sr=sr, n_mels=128)

log_h = librosa.amplitude_to_db(melspec)



# percussive

melspec = librosa.feature.melspectrogram(y_percussive, sr=sr, n_mels=128)

log_p = librosa.amplitude_to_db(melspec)



# Display the log mel spectrogram of both harmonic and percussive

plt.figure(figsize=(12,6))



plt.subplot(2,1,1)

librosa.display.specshow(log_h, sr=sr, x_axis='time', y_axis='mel')

plt.title('Log mel spectrogram for female harmonic')

plt.colorbar(format='%+02.0f dB')



plt.subplot(2,1,2)

librosa.display.specshow(log_p, sr=sr, x_axis='time', y_axis='mel')

plt.title('Log mel spectrogram for female percussive')

plt.colorbar(format='%+02.0f dB')
# Lets use this one for the male 

SAMPLE_RATE = 22050

fname_f = 'recordings/' + 'english381.mp3'  

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

X = librosa.stft(y)

H, P = librosa.decompose.hpss(X)  # Both Harmonic and Percussive as spectogram 

Hmag = librosa.amplitude_to_db(H) # Get log mel-spectogram 

Pmag = librosa.amplitude_to_db(P)



# Have a listen to male harmonic 

h = librosa.istft(H)

ipd.Audio(h, rate=SAMPLE_RATE)
# Have a listen to male percussive 

p = librosa.istft(P)

ipd.Audio(p, rate=SAMPLE_RATE)
# Display the log mel spectrogram of both harmonic and percussive

plt.figure(figsize=(12,6))



plt.subplot(2,1,1)

librosa.display.specshow(Hmag, sr=sr, x_axis='time', y_axis='mel')

plt.title('Log mel spectrogram for male harmonic')

plt.colorbar(format='%+02.0f dB')



plt.subplot(2,1,2)

librosa.display.specshow(Pmag, sr=sr, x_axis='time', y_axis='mel')

plt.title('Log mel spectrogram for male percussive')

plt.colorbar(format='%+02.0f dB')
SAMPLE_RATE = 22050

fname_f = 'recordings/' + 'english381.mp3'  

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

C = librosa.feature.chroma_cqt(y=y, sr=sr)



# Make a new figure

plt.figure(figsize=(12,4))

# To make sure that the colors span the full range of chroma values, set vmin and vmax

librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

plt.title('Chromagram')

plt.colorbar()

plt.tight_layout()
SAMPLE_RATE = 22050

fname_f = 'recordings/' + 'english385.mp3'  

y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

C = librosa.feature.chroma_cqt(y=y, sr=sr)



# Make a new figure

plt.figure(figsize=(12,4))

librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

plt.title('Chromagram')

plt.colorbar()

plt.tight_layout()