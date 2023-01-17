import os
#os.path helps us to get the absolute path 
from os.path import isdir,join
from pathlib import Path
#pathlib is the module that creates object oriented path class for different platform, and the path submodule from the pathlib cretes the concrete path if we are not sure which module is right for our platform

import pandas as pd
import numpy as np
from scipy.fftpack import fft
#fft in the scipy returns the fourier transform of the sequence
from scipy import signal
#the signal module is used to carry out different operations on the signals
from scipy.io import wavfile
#scipy is the module that is used to write file in different formats.

import librosa
#it is a package for video and music analysis
from sklearn.decomposition import PCA

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

%matplotlib inline
# a spectrogram is visual representation of the spectrum of frequency of sound or other signal
#MEL-Frequency Cepstral Coefficients(they are basically 
#non linear spectrum of a spectrum)
train_audio_path = '../input/train/audio/'
filename = '/yes/0a7c2a8d_nohash_0.wav'
sample_rate, sample = wavfile.read(str(train_audio_path)+filename)

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
freqs, times, spectrogram = log_specgram(sample, sample_rate)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(sample), sample_rate), sample)

ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + filename)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')
mean = np.mean(spectrogram, axis=0)
std = np.std(spectrogram, axis=0)
spectrogram = (spectrogram - mean) / std
