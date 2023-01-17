import os

import random

import numpy as np

import pandas as pd

import scipy

import librosa

import librosa.display

from IPython.display import Audio

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

%matplotlib inline
# Hide unnecessary warnings when loading audio files

import warnings

warnings.filterwarnings("ignore")
audio_path = '../input/birdsong-recognition/train_audio/'

test_audio_path = '../input/birdsong-recognition/example_test_audio/'
subFolderList = []

for x in os.listdir(audio_path):

    if os.path.isdir(audio_path + '/' + x):

        subFolderList.append(x)



sample_audio = []

total = 0

for x in subFolderList:

    

    # get all the wave files

    all_files = [y for y in os.listdir(audio_path+x) if '.mp3' in y]

    total += len(all_files)

    

    # collect the first file from each dir

    sample_audio.append(audio_path + x + '/'+ all_files[0])

    

    # show file counts

    print(x, len(all_files), end=' | ')
print('Number of classes:', len(subFolderList))

print('  Number of files:', total)



print('\nAverage number of files per class:', round(total/len(subFolderList)))
# demo_audio = '../input/birdsong-recognition/train_audio/aldfly/XC142068.mp3'

demo_audio = '../input/birdsong-recognition/train_audio/bewwre/XC122453.mp3' #shorter clip

demo_clip, demo_sample_rate = librosa.load(demo_audio, sr=None)

print("Class:", demo_audio.split('/')[-2])

print(" File:", demo_audio)

Audio(demo_audio)
trace = [go.Scatter(

    x=np.linspace(0, demo_sample_rate/len(demo_clip), len(demo_clip)), 

    y=demo_clip

)]

layout = go.Layout(

    title = 'Waveform <br><sup>Interactive</sup>',

    yaxis = dict(title='Amplitude'),

    xaxis = dict(title='Time'),

    )

fig = go.Figure(data=trace, layout=layout)

fig.show()
# fig = plt.figure(figsize=(25,15))

fig = plt.figure(figsize=(25,12))

for i, filepath in enumerate(sample_audio[:20]):

    # plt.subplot(5,1,i+1)

    plt.subplot(5,4,i+1)

    clip, sample_rate = librosa.load(filepath, sr=None)

    plt.title(filepath.split('/')[-2])

    plt.axis('off')

    plt.plot(clip, c='black', lw=0.5)
def plot_raw_waves(label, color=None):

    same_samples = [audio_path + f'{label}/' + y for y in os.listdir(audio_path + f'{label}/')[:20]]



    fig = plt.figure(figsize=(25,12))

    fig.suptitle(label, fontsize=30, c=color)

    for i, filepath in enumerate(same_samples):

        plt.subplot(5,4,i+1)

        clip, sample_rate = librosa.load(filepath, sr=None)

        plt.axis('off')

        plt.plot(clip, c=color, lw=0.5)
plot_raw_waves('ribgul', '#FBC02D')
plot_raw_waves('casfin', '#D81B60')
plot_raw_waves('snobun', '#F4511E')
plot_raw_waves('lazbun', '#039BE5')
def log_specgram(audio, sample_rate, window_size=20,

                 step_size=10, eps=1e-10):

    nperseg = int(round(window_size * sample_rate / 1e3))

    noverlap = int(round(step_size * sample_rate / 1e3))

    freqs, times, spec = scipy.signal.spectrogram(audio,

                                    fs=sample_rate,

                                    window='hann',

                                    nperseg=nperseg,

                                    noverlap=noverlap,

                                    detrend=False)

    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
demo_freqs, demo_times, demo_spec = log_specgram(demo_clip, sample_rate)
trace = [go.Heatmap(

    x= demo_times,

    y= demo_freqs,

    z= demo_spec.T,

    colorscale='viridis',

    )]

layout = go.Layout(

    title = 'Spectrogram <br><sup>Interactive</sup>',

    yaxis = dict(title='Frequency'),

    xaxis = dict(title='Time'),

    )

fig = go.Figure(data=trace, layout=layout)

fig.show()
fig = plt.figure(figsize=(25,12))

for i, filepath in enumerate(sample_audio[:20]):

    plt.subplot(5,4,i+1)

    label = filepath.split('/')[-2]

    plt.title(label)

    clip, sample_rate = librosa.load(filepath, sr=None)

    _, _, spectrogram = log_specgram(clip, sample_rate)

    plt.imshow(spectrogram.T, aspect='auto', origin='lower')

    plt.axis('off');
def plot_spectrograms(label):

    same_samples = [audio_path + f'{label}/' + y for y in os.listdir(audio_path + f'{label}/')[:20]]



    fig = plt.figure(figsize=(25,12))

    fig.suptitle(label, fontsize=30)

    for i, filepath in enumerate(same_samples):

        plt.subplot(5,4,i+1)

        clip, sample_rate = librosa.load(filepath, sr=None)

        _, _, spectrogram = log_specgram(clip, sample_rate)

        plt.imshow(spectrogram.T, aspect='auto', origin='lower')

        plt.axis('off')
plot_spectrograms('ribgul')
plot_spectrograms('casfin')
plot_spectrograms('snobun')
plot_spectrograms('lazbun')
trace = [go.Surface(

    x= demo_times,

    y= demo_freqs,

    z= demo_spec.T,

    colorscale='viridis',

)]

layout = go.Layout(

title='3D Specgtrogram <br><sup>Interactive</sup>',

scene = dict(

    yaxis = dict(title='Frequency', range=[demo_freqs.min(),demo_freqs .max()]),

    xaxis = dict(title='Time', range=[demo_times.min(),demo_times.max()],),

    zaxis = dict(title='Log amplitude'),

    ),

)

fig = go.Figure(data=trace, layout=layout)

fig.show()
from scipy.fftpack import fft

def custom_fft(y, fs):

    T = 1.0 / fs

    N = y.shape[0]

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    vals = 2.0/N * np.abs(yf[0:N//2])  

    return xf, vals
demo_xf, demo_vals = custom_fft(demo_clip, demo_sample_rate)
trace = [go.Scatter(

    x=demo_xf, 

    y=demo_vals,

    line_color='deeppink'

)]

layout = go.Layout(

    title = 'Fast Fourier Transform (FFT) <br><sup>Interactive</sup>',

    yaxis = dict(title='Magnitude'),

    xaxis = dict(title='Frequency'),

    )

fig = go.Figure(data=trace, layout=layout)

fig.show()
fig = plt.figure(figsize=(25,12))

for i, filepath in enumerate(sample_audio[:20]):

    plt.subplot(5,4,i+1)

    label = filepath.split('/')[-2]

    plt.title(label)

    clip, sample_rate = librosa.load(filepath, sr=None)

    xf, vals = custom_fft(clip, sample_rate)

    plt.plot(xf, vals, c='black')

    plt.axis('off');
def plot_ffts(label, color=None):

    same_samples = [audio_path + f'{label}/' + y for y in os.listdir(audio_path + f'{label}/')[:20]]



    fig = plt.figure(figsize=(25,12))

    fig.suptitle(label, fontsize=30, c=color)

    for i, filepath in enumerate(same_samples):

        plt.subplot(5,4,i+1)

        clip, sample_rate = librosa.load(filepath, sr=None)

        xf, vals = custom_fft(clip, sample_rate)

        plt.plot(xf, vals, c=color)

        plt.axis('off')
plot_ffts('ribgul', '#FBC02D')
plot_ffts('casfin', '#D81B60')
plot_ffts('snobun', '#F4511E')
plot_ffts('lazbun', '#039BE5')
demo_S = librosa.feature.melspectrogram(demo_clip, sr=demo_sample_rate, n_mels=128)    

demo_log_S = librosa.power_to_db(demo_S, ref=np.max)
trace = [go.Heatmap(

    x= demo_times,

    y= demo_freqs,

    z= demo_log_S,

    colorscale='magma',

    )]

layout = go.Layout(

    title = 'Mel Power Spectrogram <br><sup>Interactive</sup>',

    yaxis = dict(title='Mel'),

    xaxis = dict(title='Time'),

    )

fig = go.Figure(data=trace, layout=layout)

fig.show()
fig = plt.figure(figsize=(25,12))

for i, filepath in enumerate(sample_audio[:20]):

    plt.subplot(5,4,i+1)

    label = filepath.split('/')[-2]

    plt.title(label)

    clip, sample_rate = librosa.load(filepath, sr=None)

    S = librosa.feature.melspectrogram(clip, sr=sample_rate, n_mels=128)    

    log_S = librosa.power_to_db(S, ref=np.max)    

    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')

    plt.axis('off');
def plot_mel_spectrograms(label):

    same_samples = [audio_path + f'{label}/' + y for y in os.listdir(audio_path + f'{label}/')[:20]]



    fig = plt.figure(figsize=(25,12))

    fig.suptitle(label, fontsize=30)

    for i, filepath in enumerate(same_samples):

        plt.subplot(5,4,i+1)

        clip, sample_rate = librosa.load(filepath, sr=None)

        S = librosa.feature.melspectrogram(clip, sr=sample_rate, n_mels=128)    

        log_S = librosa.power_to_db(S, ref=np.max)    

        librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')

        plt.axis('off');
plot_mel_spectrograms('ribgul')
plot_mel_spectrograms('casfin')
plot_mel_spectrograms('snobun')
plot_mel_spectrograms('lazbun')
demo_mfcc = librosa.feature.mfcc(S=demo_log_S, n_mfcc=13)

demo_delta2_mfcc = librosa.feature.delta(demo_mfcc, order=2)
trace = [go.Heatmap(

    x= demo_times,

    y= demo_freqs,

    z= demo_delta2_mfcc,

    colorscale='RdBu_r',

    )]

layout = go.Layout(

    title = 'MFCC  <br><sup>Interactive</sup>',

    yaxis = dict(title='MFCC coeffs'),

    xaxis = dict(title='Time'),

    )

fig = go.Figure(data=trace, layout=layout)

fig.show()
fig = plt.figure(figsize=(25,12))

for i, filepath in enumerate(sample_audio[:20]):

    plt.subplot(5,4,i+1)

    label = filepath.split('/')[-2]

    plt.title(label)

    clip, sample_rate = librosa.load(filepath, sr=None)

    S = librosa.feature.melspectrogram(clip, sr=sample_rate, n_mels=128)    

    log_S = librosa.power_to_db(S, ref=np.max)    

    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    librosa.display.specshow(delta2_mfcc, cmap='bwr')

    plt.axis('off');
def plot_mmfcs(label):

    same_samples = [audio_path + f'{label}/' + y for y in os.listdir(audio_path + f'{label}/')[:20]]



    fig = plt.figure(figsize=(25,12))

    fig.suptitle(label, fontsize=30)

    for i, filepath in enumerate(same_samples):

        plt.subplot(5,4,i+1)

        clip, sample_rate = librosa.load(filepath, sr=None)

        S = librosa.feature.melspectrogram(clip, sr=sample_rate, n_mels=128)    

        log_S = librosa.power_to_db(S, ref=np.max)    

        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        librosa.display.specshow(delta2_mfcc, cmap='bwr')

        plt.axis('off');
plot_mmfcs('ribgul')
plot_mmfcs('casfin')
plot_mmfcs('snobun')
plot_mmfcs('lazbun')
def audio_plots(filename):

    clip, sample_rate = librosa.load(filename, sr=None)

    freqs, times, spectrogram = log_specgram(clip, sample_rate)

    label = filename.split('/')[-2]

    

    # # Normalize (if needed)

    # mean = np.mean(spectrogram, axis=0)

    # std = np.std(spectrogram, axis=0)

    # spectrogram = (spectrogram - mean) / std



    fig = plt.figure(figsize=(20,15), dpi=200)

      

    # Raw wave

    ax1 = fig.add_subplot(511)

    ax1.set_title('Raw wave of ' + label)

    ax1.set_ylabel('Amplitude')

    librosa.display.waveplot(clip.astype('float'), sr=sample_rate)



    # FFT    

    ax2 = fig.add_subplot(512)

    xf, vals = custom_fft(clip, sample_rate)

    ax2.set_title('FFT of ' + label + ' with ' + str(sample_rate) + ' Hz')

    ax2.plot(xf, vals, 'm')

    ax2.set_xlabel('Frequency')

    ax2.set_ylabel('Magnitude')

    ax2.grid()

    

    # Spectrogram    

    ax3 = fig.add_subplot(513)

    ax3.set_title('Spectrogram of ' + label)

    ax3.set_ylabel('Freqs in Hz')

    ax3.set_xlabel('Seconds')

    ax3.imshow(spectrogram.T, aspect='auto', origin='lower', 

               extent=[times.min(), times.max(), freqs.min(), freqs.max()])

    

    # Mel power spectrogram

    ax4 = fig.add_subplot(514)    

    S = librosa.feature.melspectrogram(clip, sr=sample_rate, n_mels=128)    

    log_S = librosa.power_to_db(S, ref=np.max) 

    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')

    ax4.set_title('Mel power spectrogram of ' + label)   

    

    # MFCC

    ax5 = fig.add_subplot(515)

    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    librosa.display.specshow(delta2_mfcc, cmap='bwr')

    ax5.set_title('MFCC of ' + label)

    ax5.set_ylabel('MFCC coeffs')

    ax5.set_xlabel('Time')



    plt.tight_layout()

    print(f"Class: {label}\n File: {filename}, ")

    return Audio(filename)
audio_plots(demo_audio)
audio_plots(sample_audio[1])
audio_plots(sample_audio[50])
audio_plots(sample_audio[100])
audio_plots(sample_audio[150])
audio_plots(sample_audio[200])
audio_plots(sample_audio[250])
def load_audio_file(filename):

    data, sr = librosa.load(filename, sr=None)

    return data, sr



sns.set(style='darkgrid')

def plot_time_series(data, sr, alpha=0.5, color=None, label=None):

#     plt.figure(figsize=(20,5))

    plt.title('Raw wave')

    plt.ylabel('Amplitude')

    plt.plot(np.linspace(0, sr/len(data), len(data)), data, alpha=alpha, c=color, label=label)

    plt.legend(loc='lower left', ncol=2, frameon=False)

    return Audio(data, rate=sr)



def plot_spectrogram(data, sr, label=None):

    S = librosa.feature.melspectrogram(data, sr, n_mels=128)    

    log_S = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(20,3), dpi=200)

    plt.title('Mel power spectrogram')

    plt.text(0.05, 200, label, fontsize=12, c='w')

    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')    

#     plt.axis('off')
org_data, sr = load_audio_file(demo_audio)

print("Class:", demo_audio.split('/')[-2])

print(" File:", demo_audio)



plt.figure(figsize=(20,5))

plot_time_series(org_data, sr, 0.67, 'blue', 'Actual')
plot_spectrogram(org_data, sr, 'Actual')
# Use mean & std of individual data or of all data as custom params

def normalize(X, mean=None, std=None):

    mean = mean or X.mean()

    std = std or (X-X.mean()).std()

    return ((X - mean)/std).astype(np.float64)
data = normalize(org_data)
plt.figure(figsize=(20,5))

plot_time_series(org_data, sr, 0.7, 'blue', 'Actual')

plot_time_series(data, sr, 0.6, 'red', 'Normalized')
plot_spectrogram(org_data, sr, 'Actual')

plot_spectrogram(data, sr, 'Normalized')
wn = np.random.randn(len(data))

data_wn = data + 0.1 * wn
plt.figure(figsize=(20,5))

plot_time_series(data, sr, 0.7, 'blue', 'Actual')

plot_time_series(data_wn, sr, 0.6, 'red', 'White Noise')
plot_spectrogram(data, sr, 'Actual')

plot_spectrogram(data_wn, sr, 'White Noise')
data_roll = np.roll(data, 5000)
plt.figure(figsize=(20,5))

plot_time_series(data, sr, 0.5, 'blue', 'Actual')

plot_time_series(data_roll, sr, 0.7, 'red', 'Shifted')
plot_spectrogram(data, sr, 'Actual')

plot_spectrogram(data_roll, sr, 'Shifted')
# larger value

data_stretch = librosa.effects.time_stretch(data, 1.2)
plt.figure(figsize=(20,5))

plot_time_series(data, sr, 0.3, 'blue', 'Actual')

plot_time_series(data_stretch, sr, 0.8, 'red', 'Stretched')
plot_spectrogram(data, sr, 'Actual')

plot_spectrogram(data_stretch, sr, 'Stretched')
# smaller value

data_stretch = librosa.effects.time_stretch(data, 0.8)
plt.figure(figsize=(20,5))

plot_time_series(data, sr, 0.3, 'blue', 'Actual')

plot_time_series(data_stretch, sr, 0.8, 'red', 'Stretched')
plot_spectrogram(data, sr, 'Actual')

plot_spectrogram(data_stretch, sr, 'Stretched')
data_pitch = librosa.effects.pitch_shift(data, sr, n_steps=-10)
plt.figure(figsize=(20,5))

plot_time_series(data, sr, 0.5, 'blue', 'Actual')

plot_time_series(data_pitch, sr, 0.8, 'red', 'Pitch changed')
plot_spectrogram(data, sr, 'Actual')

plot_spectrogram(data_pitch, sr, 'Pitch changed')
data_invert = -data
plt.figure(figsize=(20,5))

plot_time_series(data, sr, 1, 'blue', 'Actual')

plot_time_series(data_invert, sr, 0.7, 'red', 'Inverted')
plot_spectrogram(data, sr, 'Actual')

plot_spectrogram(data_invert, sr, 'Inverted')
x = np.linspace(0, sr/len(data), len(data))

trace = [

    go.Scatter(x=x, y=org_data, name='Actual', opacity=0.5),

    go.Scatter(x=x, y=data, name='Normalized', opacity=0.5),

    go.Scatter(x=x, y=data_wn, name='Noise', opacity=0.5),

    go.Scatter(x=x, y=data_roll, name='Shift', opacity=0.5),

    go.Scatter(x=x, y=data_stretch, name='Stretch', opacity=0.5),

    go.Scatter(x=x, y=data_pitch, name='Pitch', opacity=0.5),

    go.Scatter(x=x, y=data_invert, name='Invert', opacity=0.5),

]



layout = go.Layout(

    title = 'Augmentations <br><sup>Interactive</sup>',

    yaxis = dict(title='Amplitude'),

    xaxis = dict(title='Time'),

    legend_title_text='Augmentations <br><sup>Toggle augmentations on/off</sup>'

    )



fig = go.Figure(data=trace, layout=layout)

fig.show()
print("ACTUAL:")

org_data, sr = load_audio_file(demo_audio)

Audio(org_data, rate=sr)
data = normalize(org_data)

data_wn = data + 0.1 * wn

data_roll = np.roll(data_wn, 5000)

data_stretch = librosa.effects.time_stretch(data_roll, 1)

data_pitch = librosa.effects.pitch_shift(data_stretch, sr, n_steps=-10)

data_invert = -data_pitch

data_aug = data_invert



plt.figure(figsize=(20,5))

plot_time_series(data, sr, 0.3, 'blue', 'Actual')

print("AUGMENTED:")

plot_time_series(data_aug, sr, 0.7, 'red', 'Augmented')
plot_spectrogram(data, sr, 'Actual')

plot_spectrogram(data_aug, sr, 'Augmented')
clip, sample_rate = librosa.load(demo_audio, sr=None)

print("Class:", demo_audio.split('/')[-2])

print(" File:", demo_audio)

Audio(demo_audio)
S = librosa.feature.melspectrogram(clip, sr=sample_rate, n_mels=256)    

log_S = librosa.power_to_db(S, ref=np.max)



plt.figure(figsize=(20,4))

plt.title('Mel power spectrogram')

plt.text(0.05, 200, 'original audio', fontsize=12, c='w')

librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel');
def spec_augment(spec: np.ndarray, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):

    spec = spec.copy()

    for i in range(num_mask):

        num_freqs, num_frames = spec.shape

        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        

        num_freqs_to_mask = int(freq_percentage * num_freqs)

        num_frames_to_mask = int(time_percentage * num_frames)

        

        t0 = int(np.random.uniform(low=0.0, high=num_frames - num_frames_to_mask))

        f0 = int(np.random.uniform(low=0.0, high=num_freqs - num_freqs_to_mask))

        

        spec[:, t0:t0 + num_frames_to_mask] = 0     

        spec[f0:f0 + num_freqs_to_mask, :] = 0 

        

    return spec
for i in range(4):

    plt.figure(figsize=(20,3))

    librosa.display.specshow(spec_augment(log_S), sr=sample_rate, x_axis='time', y_axis='mel');

#     plt.axis('off')
def pad_to_size(signal, size, mode):

    if signal.shape[1] < size:

        padding = size - signal.shape[1]

        offset = padding // 2

        pad_width = ((0, 0), (offset, padding - offset))

        if mode == 'constant':

            signal = np.pad(signal, pad_width, 'constant', constant_values=signal.min())

        elif mode == 'wrap':

            signal = np.pad(signal, pad_width, 'wrap')

    return signal    
img_size = 400



plt.figure(figsize=(20,4))

plt.title('Mel power spectrogram')

plt.text(0.05, 200, 'constant', fontsize=12, c='w')

librosa.display.specshow(pad_to_size(log_S, img_size, 'constant'), sr=sample_rate, x_axis='time', y_axis='mel');



plt.figure(figsize=(20,4))

plt.title('Mel power spectrogram')

plt.text(0.05, 200, 'wrap', fontsize=12, c='w')

librosa.display.specshow(pad_to_size(log_S, img_size, 'wrap'), sr=sample_rate, x_axis='time', y_axis='mel');