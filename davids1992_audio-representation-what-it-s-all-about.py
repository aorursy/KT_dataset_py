import os

from os.path import isdir, join

from pathlib import Path

import pandas as pd



# Math

import numpy as np

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile

import librosa



from sklearn.decomposition import PCA



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import IPython.display as ipd

import librosa.display



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import pandas as pd



%matplotlib inline
train_audio_path = '../input/train_curated/' 

# 8a8110c2 c2aff189 d7d25898 0a2895b8 6459fc05 54940c5c 024e0fbe c6f8f09e f46cc65b  

# 1acaf122 a0a85eae da3a5cd5 412c28dd 0f301184 2ce5262c

sample_rate, samples1 = wavfile.read(os.path.join(train_audio_path, '98b0df76.wav'))

sample_rate, samples2 = wavfile.read(os.path.join(train_audio_path, 'd7d25898.wav'))
ipd.Audio(samples1, rate=sample_rate)
ipd.Audio(samples2, rate=sample_rate)
def plot_raw_wave(samples):

    plt.figure(figsize=(14, 3))

    plt.title('Raw wave')

    plt.ylabel('Amplitude')

    # ax1.plot(np.linspace(0, sample_rate/len(samples1), sample_rate), samples1)

    plt.plot(samples)

    plt.show()

plot_raw_wave(samples1)

plot_raw_wave(samples2)
def custom_fft(y, fs):

    T = 1.0 / fs

    N = y.shape[0]

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half

    # FFT is also complex, to we take just the real part (abs)

    return xf, vals



def plot_custom_fft(samples, sample_rate):

    xf, vals = custom_fft(samples, sample_rate)

    plt.figure(figsize=(12, 4))

    plt.title('FFT of recording sampled with ' + str(sample_rate) + ' Hz')

    plt.plot(xf, vals)

    plt.xlabel('Frequency')

    plt.grid()

    plt.show()
plot_custom_fft(samples1, sample_rate)

plot_custom_fft(samples2, sample_rate)
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



def plot_log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):

    

    fig = plt.figure(figsize=(14, 3))

    freqs, times, spectrogram = log_specgram(audio, sample_rate)

    plt.imshow(spectrogram.T, aspect='auto', origin='lower', 

               extent=[times.min(), times.max(), freqs.min(), freqs.max()])

    plt.yticks(freqs[::16])

    plt.xticks(times[::16])

    plt.title('Spectrogram')

    plt.ylabel('Freqs in Hz')

    plt.xlabel('Seconds')

    plt.show()
plot_log_specgram(samples1, sample_rate)

plot_log_specgram(samples2, sample_rate)
# From this tutorial

# https://github.com/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb

S = librosa.feature.melspectrogram(samples1.astype(float), sr=sample_rate, n_mels=128)



# Convert to log scale (dB). We'll use the peak power (max) as reference.

log_S = librosa.power_to_db(S, ref=np.max)



plt.figure(figsize=(12, 4))

librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')

plt.title('Mel power spectrogram ')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)



# Let's pad on the first and second deltas while we're at it

delta2_mfcc = librosa.feature.delta(mfcc, order=2)



plt.figure(figsize=(12, 4))

librosa.display.specshow(delta2_mfcc)

plt.ylabel('MFCC coeffs')

plt.xlabel('Time')

plt.title('MFCC')

plt.colorbar()

plt.tight_layout()
freqs, times, spectrogram = log_specgram(samples2, sample_rate)

data = [go.Surface(z=spectrogram.T)]

layout = go.Layout(

    title='Specgtrogram of "yes" in 3d',

#     scene = dict(

#     yaxis = dict(title='Frequencies', range=freqs),

#     xaxis = dict(title='Time', range=times),

#     zaxis = dict(title='Log amplitude'),

#     ),

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)