# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
x, fs = librosa.load('../input/s0302b_1.wav')
t = np.linspace(0, len(x)/fs, num=len(x))
plt.figure()

plt.title('Audio wave')

plt.plot(t, x)

plt.xlabel('time/sec')

plt.show()
x_stft = librosa.stft(x)
librosa.display.specshow(librosa.amplitude_to_db(x_stft, ref=np.max), y_axis='log', x_axis='time')

plt.title('Power spectrogram')

plt.colorbar(format='%+2.0f dB')

plt.tight_layout()