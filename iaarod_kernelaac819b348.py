import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import librosa

import librosa.display

import numpy as np



import matplotlib

import matplotlib.pyplot as plt;



set_a_dir = "../input/heartbeat-sounds/set_a/"

set_b_dir = "../input/heartbeat-sounds/set_b/"

set_a = os.listdir(set_a_dir)

set_b = os.listdir(set_b_dir)
# Load Wav File

n = 1

x, samplerate = librosa.load(set_b_dir + set_b[n], sr=None)
# Display Waveform

librosa.display.waveplot(x, sr=samplerate)
import IPython.display as ipd

ipd.Audio(set_b_dir + set_b[n])
# Plot Spectogram

plt.figure(figsize=(10,10))

spect = librosa.stft(x)

spect_db = librosa.amplitude_to_db(abs(spect))

librosa.display.specshow(spect_db, sr=samplerate)
librosa.feature.mfcc(y=x, sr=samplerate)