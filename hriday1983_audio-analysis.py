# import relevant modules

import numpy as np # multidimensional array

import pandas as pd # panel data

import matplotlib.pyplot as plt # data visualisation

from glob import glob # reading files





import librosa as lr # audio analysis



# Read the audio file and create time array

audio, sfreq = lr.load('../input/sample.wav')

time = np.arange(0, len(audio))/sfreq



# plot the audio file over time

fig, ax = plt.subplots(figsize=(12,4))

ax.plot(time,audio)

ax.set(xlabel = 'Time (s)', ylabel = 'Sound Amplitute')

plt.show()