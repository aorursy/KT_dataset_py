



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



train_audio_path = "../input/train/audio/"

filename = '../input/train/audio/cat/004ae714_nohash_0.wav'

sample_rate,samples = wavfile.read(filename)
# defining a function that calculates the spectrograms



def specgram(audio, sample_rate, ):

    window_size=20 

    step_size = 10

    esp = 1e-10

    nperseg = int(round(window_size*sample_rate/1e3))

    noverlap = int(round(step_size*sample_rate/1e3))

    

    freqs, times, spec = signal.spectrogram(audio,

                                    fs=sample_rate,

                                    window='hann',

                                    nperseg=nperseg,

                                    noverlap=noverlap,

                                    detrend=False)

    return freqs, times, np.log(spec.T.astype(np.float32) + 1e-10)
freqs, times, spectrogram = specgram(samples, sample_rate)

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
mean = np.mean(spectrogram, axis=0)

std = np.std(spectrogram, axis=0)

spectrogram = (spectrogram - mean) / std
spectrogram
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

    marker=dict(color = number_of_recordings, colorscale='Viridius', showscale=True

    ),

)

layout = go.Layout(

    title='Number of recordings in given label',

    xaxis = dict(title='Words'),

    yaxis = dict(title='Number of recordings')

)

py.iplot(go.Figure(data=[trace], layout=layout))
def violinplot_frequency(dirs, freq_ind):

    """ Plot violinplots for given words (waves in dirs) and frequency freq_ind

    from all frequencies freqs."""



    spec_all = []  # Contain spectrograms

    ind = 0

    for direct in dirs:

        spec_all.append([])



        waves = [f for f in os.listdir(join(train_audio_path, direct)) if

                 f.endswith('.wav')]

        for wav in waves[:100]:

            sample_rate, samples = wavfile.read(

                train_audio_path + direct + '/' + wav)

            freqs, times, spec = specgram(samples, sample_rate)

            spec_all[ind].extend(spec[:, freq_ind])

        ind += 1



    # Different lengths = different num of frames. Make number equal

    minimum = min([len(spec) for spec in spec_all])

    spec_all = np.array([spec[:minimum] for spec in spec_all])



    plt.figure(figsize=(13,7))

    plt.title('Frequency ' + str(freqs[freq_ind]) + ' Hz')

    plt.ylabel('Amount of frequency in a word')

    plt.xlabel('Words')

    sns.violinplot(data=pd.DataFrame(spec_all.T, columns=dirs))

    plt.show()
violinplot_frequency(dirs, 120)

annon = 0

for i in os.listdir(str(train_audio_path)+"go/"):

    sample_rate,samples = wavfile.read(str(train_audio_path)+"go/"+str(i))

    if sample_rate != samples.shape[0]:

        annon+=1



print(annon)

print(len(os.listdir(str(train_audio_path)+"go/")))

print(len(samples))

    
# defining a fast furiour transformation

def custom_fft(y, fs):

    T = 1.0 / fs

    N = y.shape[0]

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half

    # FFT is also complex, to we take just the real part (abs)

    return xf, vals
# fft_all = []

# names = []

# for direct in dirs:

#     waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]

#     for wav in waves:

#         sample_rate, samples = wavfile.read(train_audio_path + direct + '/' + wav)

#         if samples.shape[0] != sample_rate:

#             samples = np.append(samples, np.zeros(abs(sample_rate - samples.shape[0] )))

#         x, val = custom_fft(samples, sample_rate)

#         print(x, val)

#         fft_all.append(val)

#         names.append(direct + '/' + wav)



# fft_all = np.array(fft_all)



# # Normalization

# fft_all = (fft_all - np.mean(fft_all)) / np.std(fft_all)



# # Dim reduction

# pca = PCA(n_components=3)

# fft_all = pca.fit_transform(fft_all)



# def interactive_3d_plot(data, names):

#     scatt = go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode='markers', text=names)

#     data = go.Data([scatt])

#     layout = go.Layout(title="Anomaly detection")

#     figure = go.Figure(data=data, layout=layout)

#     py.iplot(figure)

    

# interactive_3d_plot(fft_all, names)

            
# print(fft_all.shape)