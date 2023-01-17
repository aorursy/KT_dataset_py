# Import our libraries

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import specgram

import pandas as pd

import os

import IPython.display as ipd  # To play sound in the notebook
# Source - RAVDESS; Gender - Female; Emotion - Angry 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_08/03-01-05-02-01-01-08.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)



# audio wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.waveplot(X, sr=sample_rate)

plt.title('Audio sampled at 44100 hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
# Source - RAVDESS; Gender - Male; Emotion - Angry 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)



# audio wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.waveplot(X, sr=sample_rate)

plt.title('Audio sampled at 44100 hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
# Source - RAVDESS; Gender - Female; Emotion - Happy 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_12/03-01-03-01-02-01-12.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)



# audio wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.waveplot(X, sr=sample_rate)

plt.title('Audio sampled at 44100 hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
# Source - RAVDESS; Gender - Male; Emotion - Happy 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_11/03-01-03-01-02-02-11.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)



# audio wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.waveplot(X, sr=sample_rate)

plt.title('Audio sampled at 44100 hrz')



# MFCC

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

librosa.display.specshow(mfcc, x_axis='time')

plt.ylabel('MFCC')

plt.colorbar()



ipd.Audio(path)
# Source - RAVDESS; Gender - Female; Emotion - Angry 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_08/03-01-05-02-01-01-08.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

print(len(female))



# Source - RAVDESS; Gender - Male; Emotion - Angry 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

male = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

male = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

print(len(male))



# audio wave

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

plt.plot(female, label='female')

plt.plot(male, label='male')

plt.legend()
# Source - RAVDESS; Gender - Female; Emotion - happy 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_12/03-01-03-01-02-01-12.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

print(len(female))



# Source - RAVDESS; Gender - Male; Emotion - happy 

path = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_11/03-01-03-01-02-02-11.wav"

X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  

male = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

male = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

print(len(male))



# Plot the two audio waves together

plt.figure(figsize=(20, 15))

plt.subplot(3,1,1)

plt.plot(female, label='female')

plt.plot(male, label='male')

plt.legend()