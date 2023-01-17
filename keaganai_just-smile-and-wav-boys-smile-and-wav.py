import os

from os.path import isdir, join

from pathlib import Path

import pandas as pd

import wave



# Math

import numpy as np

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile

import librosa

import cv2



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



import warnings

warnings.filterwarnings('ignore')
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

    

class EasyDict(dict):

    def __init__(self, d=None, **kwargs):

        if d is None:

            d = {}

        if kwargs:

            d.update(**kwargs)

        for k, v in d.items():

            setattr(self, k, v)

        # Class attributes

        for k in self.__class__.__dict__.keys():

            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):

                setattr(self, k, getattr(self, k))



    def __setattr__(self, name, value):

        if isinstance(value, (list, tuple)):

            value = [self.__class__(x)

                     if isinstance(x, dict) else x for x in value]

        elif isinstance(value, dict) and not isinstance(value, self.__class__):

            value = self.__class__(value)

        super(EasyDict, self).__setattr__(name, value)

        super(EasyDict, self).__setitem__(name, value)



    __setitem__ = __setattr__



    def update(self, e=None, **f):

        d = e or dict()

        d.update(f)

        for k in d:

            setattr(self, k, d[k])



    def pop(self, k, d=None):

        delattr(self, k)

        return super(EasyDict, self).pop(k, d)

    

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

    

def plot_raw_wave(samples):

    plt.figure(figsize=(14, 3))

    plt.title('Raw wave')

    plt.ylabel('Amplitude')

    # ax1.plot(np.linspace(0, sample_rate/len(samples1), sample_rate), samples1)

    plt.plot(samples)

    plt.show()

    



train_df = pd.read_csv('../input/train_curated.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

# print('train: {}'.format(train_df.shape))

# print('test: {}'.format(sample_submission.shape))



ROOT = '../input/'

test_root = os.path.join(ROOT, 'test/')

train_root = os.path.join(ROOT, 'train_curated/')





CONFIG = EasyDict()

CONFIG.hop_length = 347 # to make time steps 128

CONFIG.fmin = 20

CONFIG.fmax = 44100 / 2

CONFIG.n_fft = 480



N_SAMPLES = 48

SAMPLE_DIM = 256



TRAINING_CONFIG = {

    'sample_dim': (N_SAMPLES, SAMPLE_DIM),

    'padding_mode': cv2.BORDER_REFLECT,

}



# print(CONFIG)

# print(TRAINING_CONFIG)



train_df.head()

    

class DataProcessor(object):

    

    def __init__(self, debug=False):

        self.debug = debug

        

        # Placeholders for global statistics

        self.mel_mean = None

        self.mel_std = None

        self.mel_max = None

        self.mfcc_max = None

        

    def createMel(self, filename, params, normalize=False):

        """

        Create Mel Spectrogram sample out of raw wavfile

        """

        y, sr = librosa.load(filename, sr=None)

        mel = librosa.feature.melspectrogram(y, sr, n_mels=N_SAMPLES, **params)

        mel = librosa.power_to_db(mel)

        if normalize:

            if self.mel_mean is not None and self.mel_std is not None:

                mel = (mel - self.mel_mean) / self.mel_std

            else:

                sample_mean = np.mean(mel)

                sample_std = np.std(mel)

                mel = (mel - sample_mean) / sample_std

            if self.mel_max is not None:

                mel = mel / self.mel_max

            else:

                mel = mel / np.max(np.abs(mel))

        return mel

    

    def createMfcc(self, filename, params, normalize=False):

        """

        Create MFCC sample out of raw wavfile

        """

        y, sr = librosa.load(filename, sr=None)

        nonzero_idx = [y > 0]

        y[nonzero_idx] = np.log(y[nonzero_idx])

        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=N_SAMPLES, **params)

        if normalize:

            if self.mfcc_max is not None:

                mfcc = mfcc / self.mfcc_max

            else:

                mfcc = mfcc / np.max(np.abs(mfcc))

        return mfcc

    

    def prepareSample(self, root, row, 

                      preprocFunc, 

                      preprocParams, trainingParams, 

                      test_mode=False, normalize=False, 

                      proc_mode='split'):

        """

        Prepare sample for model training.

        Function takes row of DataFrame, extracts filename and labels and processes them.

        

        If proc_mode is 'split':

        Outputs sets of arrays of constant shape padded to TRAINING_CONFIG shape

        with selected padding mode, also specified in TRAINING_CONFIG.

        This approach prevents loss of information caused by trimming the audio sample,

        instead it splits it into equally-sized parts and pads them.

        To account for creation of multiple samples, number of labels are multiplied to a number

        equal to number of created samples.

        

        If proc_mode is 'resize':

        Resizes the original processed sample to (SAMPLE_DIM, N_SAMPLES) shape.

        """

        

        assert proc_mode in ['split', 'resize'], 'proc_must be one of split or resize'

        

        filename = os.path.join(root, row['fname'])

        if not test_mode:

            labels = row['labels']

            

        sample = preprocFunc(filename, preprocParams, normalize=normalize)

        # print(sample.min(), sample.max())

        

        if proc_mode == 'split':

            sample_split = np.array_split(

                sample, np.ceil(sample.shape[1] / SAMPLE_DIM), axis=1)

            samples_pad = []

            for i in sample_split:

                padding_dim = SAMPLE_DIM - i.shape[1]

                sample_pad = cv2.copyMakeBorder(i, 0, 0, 0, padding_dim, trainingParams['padding_mode'])

                samples_pad.append(sample_pad)

            samples_pad = np.asarray(samples_pad)

            if not test_mode:

                labels = [labels] * len(samples_pad)

                labels = np.asarray(labels)

                return samples_pad, labels

            return samples_pad

        elif proc_mode == 'resize':

            sample_pad = cv2.resize(sample, (SAMPLE_DIM, N_SAMPLES), interpolation=cv2.INTER_NEAREST)

            sample_pad = np.expand_dims(sample_pad, axis=0)

            if not test_mode:

                labels = np.asarray(labels)

                return sample_pad, labels

            return sample_pad

        

    

processor = DataProcessor()



def spec_augment(spec: np.ndarray, num_mask=2, 

                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):



    spec = spec.copy()

    for i in range(num_mask):

        all_frames_num, all_freqs_num = spec.shape

        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)

        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)

        f0 = int(f0)

        spec[:, f0:f0 + num_freqs_to_mask] = 0



        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        

        num_frames_to_mask = int(time_percentage * all_frames_num)

        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)

        t0 = int(t0)

        spec[t0:t0 + num_frames_to_mask, :] = 0

    

    return spec

    

audio_path = os.path.join("../input/train_curated/d7d25898.wav")

sr, audio = wavfile.read(audio_path)



x = librosa.feature.melspectrogram(y=audio.astype(float), sr=sr, S=None, n_fft=512, hop_length=256, n_mels=40).T

x = librosa.power_to_db(x, ref=np.max)



train = pd.read_csv("../input/train_curated.csv")

train_new = train.sort_values('labels').reset_index()

train_new['nframes'] = train_new['fname'].apply(lambda f: wave.open('../input/train_curated/' + f).getnframes())



train_fname = train_new.head(1000)

path = "../input/train_curated/"



FILENAME  = train_root + train_df.fname[5]

NORMALIZE = True



sample_mel = processor.createMel(FILENAME, CONFIG, normalize=NORMALIZE)

sample_mfcc = processor.createMfcc(FILENAME, CONFIG, normalize=NORMALIZE)

# print(sample_mel.shape)

# print(sample_mfcc.shape)



sample_idxs = np.random.choice(np.arange(0, len(train_df)), 5)



train_audio_path = '../input/train_curated/' 

# 8a8110c2 c2aff189 d7d25898 0a2895b8 6459fc05 54940c5c 024e0fbe c6f8f09e f46cc65b  

# 1acaf122 a0a85eae da3a5cd5 412c28dd 0f301184 2ce5262c

sample_rate, samples1 = wavfile.read(os.path.join(train_audio_path, '98b0df76.wav'))

sample_rate, samples2 = wavfile.read(os.path.join(train_audio_path, 'd7d25898.wav'))
ipd.Audio(samples1, rate=sample_rate)
plot_raw_wave(samples1)

plot_raw_wave(samples2)

ipd.Audio(samples2, rate=sample_rate)
fig, axes = plt.subplots(figsize=(16,5))

train_new.nframes.hist(bins=100)

plt.suptitle('Frame Length Distribution in Train Curated', ha='center', fontsize='large');
for i in sample_idxs:

    sample_prep, labels = processor.prepareSample(

        train_root, 

        train_df.iloc[i, :], 

        processor.createMel,  

        CONFIG, TRAINING_CONFIG,  

        test_mode=False,

        proc_mode='split',  

    )  

    NCOLS = 2

    NROWS = int(np.ceil(sample_prep.shape[0] / NCOLS))

    fig, ax = plt.subplots(NCOLS, NROWS, figsize=(20, 5))

    fig.suptitle('Sample: {}'.format(i))

    idx = 0

    for c in range(NCOLS):

        if NROWS > 1:

            for r in range(NROWS):

                if idx < sample_prep.shape[0]:

                    ax[c, r].imshow(sample_prep[idx], cmap='Spectral')

                    ax[c, r].set_title('class: {}'.format(labels[idx]))

                    idx += 1

        else:

            if idx < sample_prep.shape[0]:

                ax[c].imshow(sample_prep[idx], cmap='Spectral')

                ax[c].set_title('class: {}'.format(labels[idx]))

                idx += 1

    plt.show()

S = librosa.feature.melspectrogram(samples1.astype(float), sr=sample_rate, n_mels=128)



# Convert to log scale (dB). We'll use the peak power (max) as reference.

log_S = librosa.power_to_db(S, ref=np.max)



plt.figure(figsize=(12, 4))

librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')

plt.title('Mel power spectrogram ')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()
freqs, times, spectrogram = log_specgram(samples2, sample_rate)

data = [go.Surface(z=spectrogram.T)]

layout = go.Layout(

    title='Specgtogram 3d',

    

#     scene = dict(

#     yaxis = dict(title='Frequencies', range=freqs),

#     xaxis = dict(title='Time', range=times),

#     zaxis = dict(title='Log amplitude'),

#     )

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# 