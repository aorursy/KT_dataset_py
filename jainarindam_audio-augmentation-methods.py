import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import librosa

import matplotlib.pyplot as plt

import os

import cv2

import IPython.display as ipd

from IPython.display import Audio, IFrame, display

import plotly.graph_objects as go

import librosa

import librosa.display

plt.style.use("ggplot")

train = pd.read_csv("../input/birdsong-recognition/train.csv")

species=train.species.value_counts()

fig = go.Figure(data=[

    go.Bar(y=species.values, x=species.index,marker_color='deeppink')

])



fig.update_layout(title='Distribution of Bird Species')

fig.show()
file_path='../input/birdsong-resampled-train-audio-04/wooscj2/XC67042.wav'

x , sr = librosa.load(file_path)

librosa.display.waveplot(x, sr=sr)

Audio(x, rate=sr)
def noise(data, noise_factor):

    noise = np.random.randn(len(data))

    augmented_data = data + noise_factor * noise

    # Cast back to same data type

    augmented_data = augmented_data.astype(type(data[0]))

    return augmented_data
n=noise(x,0.01)

librosa.display.waveplot(n, sr=sr)
def shifting_time(data, sampling_rate, shift_max, shift_direction):

    shift = np.random.randint(sampling_rate * shift_max)

    if shift_direction == 'right':

        shift = -shift

    elif self.shift_direction == 'both':

        direction = np.random.randint(0, 2)

        if direction == 1:

            shift = -shift

    augmented_data = np.roll(data, shift)

    # Set to silence for heading/ tailing

    if shift > 0:

        augmented_data[:shift] = 0

    else:

        augmented_data[shift:] = 0

    return augmented_data
s=shifting_time(x,sr,1,'right')

librosa.display.waveplot(s, sr=sr)
def speed(data, speed_factor):

    return librosa.effects.time_stretch(data, speed_factor)
v=speed(x,2)

librosa.display.waveplot(v, sr=sr)
def pitch(data, sampling_rate, pitch_factor):

    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
p=pitch(x,sr,2)

librosa.display.waveplot(p, sr)