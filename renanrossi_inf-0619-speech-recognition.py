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
!pip install pyunpack

!pip install patool

os.system('apt-get install p7zip')
from pyunpack import Archive

import shutil

if not os.path.exists('/kaggle/working/train/'):

    os.makedirs('/kaggle/working/train/')

Archive('../input/tensorflow-speech-recognition-challenge/train.7z').extractall('/kaggle/working/train/')

for dirname, _, filenames in os.walk('/kaggle/working/train/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_audio_path = '../working/train/train/audio/'

filename = '/yes/0a7c2a8d_nohash_0.wav'

sample_rate, samples = wavfile.read(str(train_audio_path) + filename)
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
freqs, times, spectrogram = log_specgram(samples, sample_rate)



fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(211)

ax1.set_title('Raw wave of ' + filename)

ax1.set_ylabel('Amplitude')

ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)



ax2 = fig.add_subplot(212)

ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 

           extent=[times.min(), times.max(), freqs.min(), freqs.max()])

ax2.set_yticks(freqs[::16])

ax2.set_xticks(times[::16])

ax2.set_title('Spectrogram of ' + filename)

ax2.set_ylabel('Freqs in Hz')

ax2.set_xlabel('Seconds')
print(samples.shape)

print(samples[0])

print(sample_rate)



print(freqs.shape)

print(freqs[0])

print(times[0])

print(spectrogram.shape)
ipd.Audio(samples, rate=sample_rate)
samples_cut = samples[6000:13000]

sample_rate_cut = 7000



freqs, times, spectrogram = log_specgram(samples_cut, sample_rate)



fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(211)

ax1.set_title('Raw wave of ' + filename)

ax1.set_ylabel('Amplitude')

ax1.plot(np.linspace(0, sample_rate/len(samples_cut), sample_rate_cut), samples_cut)



ax2 = fig.add_subplot(212)

ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 

           extent=[times.min(), times.max(), freqs.min(), freqs.max()])

ax2.set_yticks(freqs[::16])

ax2.set_xticks(times[::16])

ax2.set_title('Spectrogram of ' + filename)

ax2.set_ylabel('Freqs in Hz')

ax2.set_xlabel('Seconds')
ipd.Audio(samples_cut, rate=sample_rate)
dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]

dirs.sort()

print('Number of labels: ' + str(len(dirs)))
# Calculate

number_of_recordings = []

for direct in dirs:

    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]

    number_of_recordings.append(len(waves))



# Plot

data = [go.Histogram(x=dirs, y=number_of_recordings)]

trace = go.Bar(

    x=dirs,

    y=number_of_recordings,

    marker=dict(color = number_of_recordings, colorscale='icefire', showscale=True),

)

layout = go.Layout(

    title='Number of recordings in given label',

    xaxis = dict(title='Words'),

    yaxis = dict(title='Number of recordings')

)

py.iplot(go.Figure(data=[trace], layout=layout))
def custom_fft(y, fs):

    T = 1.0 / fs

    N = y.shape[0]

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half

    # FFT is also complex, to we take just the real part (abs)

    return xf, vals
filenames = ['on/004ae714_nohash_0.wav', 'on/0137b3f4_nohash_0.wav']

for filename in filenames:

    sample_rate, samples = wavfile.read(str(train_audio_path) + filename)

    xf, vals = custom_fft(samples, sample_rate)

    plt.figure(figsize=(12, 4))

    plt.title('FFT of speaker ' + filename[4:11])

    plt.plot(xf, vals)

    plt.xlabel('Frequency')

    plt.grid()

    plt.show()
print('Speaker ' + filenames[0][4:11])

ipd.Audio(join(train_audio_path, filenames[0]))
print('Speaker ' + filenames[1][4:11])

ipd.Audio(join(train_audio_path, filenames[1]))
filename = '/yes/01bb6a2a_nohash_1.wav'

sample_rate, samples = wavfile.read(str(train_audio_path) + filename)

freqs, times, spectrogram = log_specgram(samples, sample_rate)



plt.figure(figsize=(10, 7))

plt.title('Spectrogram of ' + filename)

plt.ylabel('Freqs')

plt.xlabel('Time')

plt.imshow(spectrogram.T, aspect='auto', origin='lower', 

           extent=[times.min(), times.max(), freqs.min(), freqs.max()])

plt.yticks(freqs[::16])

plt.xticks(times[::16])

plt.show()
num_of_shorter = 0

for direct in dirs:

    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]

    for wav in waves:

        sample_rate, samples = wavfile.read(train_audio_path + direct + '/' + wav)

        if samples.shape[0] < sample_rate:

            num_of_shorter += 1

print('Number of recordings shorter than 1 second: ' + str(num_of_shorter))
to_keep = 'yes no up down left right on off stop go'.split()

dirs = [d for d in dirs if d in to_keep]



print(dirs)



for direct in dirs:

    vals_all = []

    spec_all = []



    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]

    for wav in waves:

        sample_rate, samples = wavfile.read(train_audio_path + direct + '/' + wav)

        if samples.shape[0] != 16000:

            continue

        xf, vals = custom_fft(samples, 16000)

        vals_all.append(vals)

        freqs, times, spec = log_specgram(samples, 16000)

        spec_all.append(spec)



    plt.figure(figsize=(14, 4))

    plt.subplot(121)

    plt.title('Mean fft of ' + direct)

    plt.plot(np.mean(np.array(vals_all), axis=0))

    plt.grid()

    plt.subplot(122)

    plt.title('Mean specgram of ' + direct)

    plt.imshow(np.mean(np.array(spec_all), axis=0).T, aspect='auto', origin='lower', 

               extent=[times.min(), times.max(), freqs.min(), freqs.max()])

    plt.yticks(freqs[::16])

    plt.xticks(times[::16])

    plt.show()
fft_all = []

names = []

for direct in dirs:

    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]

    for wav in waves:

        sample_rate, samples = wavfile.read(train_audio_path + direct + '/' + wav)

        if samples.shape[0] != sample_rate:

            samples = np.append(samples, np.zeros((sample_rate - samples.shape[0], )))

        x, val = custom_fft(samples, sample_rate)

        fft_all.append(val)

        names.append(direct + '/' + wav)



fft_all = np.array(fft_all)



# Normalization

fft_all = (fft_all - np.mean(fft_all, axis=0)) / np.std(fft_all, axis=0)



# Dim reduction

pca = PCA(n_components=3)

fft_all = pca.fit_transform(fft_all)



def interactive_3d_plot(data, names):

    scatt = go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode='markers', text=names)

    data = go.Data([scatt])

    layout = go.Layout(title="Anomaly detection")

    figure = go.Figure(data=data, layout=layout)

    py.iplot(figure)

    

interactive_3d_plot(fft_all, names)
print('Recording yes/5165cf0a_nohash_0.wav')

ipd.Audio(join(train_audio_path, 'yes/5165cf0a_nohash_0.wav'))
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

# define baseline model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(512, input_dim=3, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.summary()

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from keras.wrappers.scikit_learn import KerasClassifier

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
from sklearn import preprocessing

gt = []

for name in names:

    gt.append(name.split('/')[0])



le = preprocessing.LabelEncoder()

le.fit(gt)

gt = le.transform(gt)
estimator = estimator.fit(fft_all, gt)
print(estimator.history)