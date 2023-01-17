import IPython.display as ipd
import librosa as lb
import librosa.display as ld
import sklearn as sk
import seaborn as sb
import plotly as ply
import scipy as sp
from scipy import signal
from scipy.fftpack import fft
%pylab inline
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
root_path = r'..'
out_path = r'.'
model_path = r'.'
train_path = os.path.join(root_path, 'input', 'train', 'audio')
test_path = os.path.join(root_path, 'input', 'test', 'audio')
data, sampling_rate = lb.load(train_path + '/_background_noise_/dude_miaowing.wav')
print(data, sampling_rate)
import random
def log_spectrogram(audio, sampling_rate, window_size=20, step_size=10, eps=1e-10):
    nps = int(round(window_size * sampling_rate / 1e3))
    nol = int(round(step_size * sampling_rate / 1e3))
    frequencies, times, specs = signal.spectrogram(audio, fs=sampling_rate, window='hann', nperseg=nps, noverlap=nol, detrend=False)
    return frequencies, times, np.log(specs.T.astype(np.float32) + eps)
file = train_path + '/tree/0e4d22f1_nohash_0.wav'
sampling_rate, samples = sp.io.wavfile.read(file)

frequencies, times, spectrogram = log_spectrogram(samples, sampling_rate)

fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw Wave of ' + file)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sampling_rate/len(samples), sampling_rate), samples)
ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
ax2.set_yticks(frequencies[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + file)
ax2.set_ylabel('Frequencies in Hz')
ax2.set_xlabel('Seconds')
mean = np.mean(spectrogram, axis=0)
std = np.std(spectrogram, axis=0)
spectrogram = (spectrogram - mean) / std
print(spectrogram)
train_pic_path = os.path.join(root_path, 'input', 'train')
test_pic_path = os.path.join(root_path, 'input', 'test')

import matplotlib as mpl
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D, GRU
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, plot_model
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import SimpleRNN, LSTM 
from keras.layers.embeddings import Embedding

mpl.rc('font', family = 'serif', size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2
hyper_pwr = 0.5
hyper_train_ratio = 0.9
hyper_n = 25
hyper_m = 15
hyper_NR = 208
hyper_NC = 112
hyper_delta = 0.3
hyper_dropout0 = 0.2
hyper_dropout1 = 0.4
hyper_dropout2 = 0.6
hyper_dropout3 = 0.6
hyper_dropout4 = 0.4
hyper_dropout5 = 0.7

target_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
L = 16000
def load_audio_data(path, ltoi):
    x = []
    y = []
    for i, folder in enumerate(os.listdir(path)):
        for filename in os.listdir(path + '/' + folder):
            if filename == 'README.md':
                continue
            rate, sample = wavfile.read(train_path + '/' + folder + '/' + filename)
            assert(rate == L)
            if folder == '_background_noise_':
                length = len(sample)
                for j in range(int(length/rate)):
                    x.append(np.array(sample[j*rate: (j+1)*rate]))
                    y.append(ltoi['silence'])
            else:
                x.append(np.array(sample))
                label = folder
                if folder not in target_labels:
                    label = 'unknown'
                y.append(ltoi[label])
    x = np.array(pad_sequences(x, maxlen=L))
    y = np.array(y)
    df = pd.DataFrame()
    df['x'] = list(x)
    df['y'] = list(y)
    return df
import os
os.listdir('{0}/_background_noise_'.format(train_path))
from scipy.io import wavfile
print("LOADING RAW DATA!")
label2idx = {}
idmap = {}
for i,lab in enumerate(target_labels):
    label2idx[lab] = i
    idmap[i] = lab
raw_df = load_audio_data(train_path, label2idx)
print(label2idx)
print(idmap)
print(raw_df.x.as_matrix().shape)
print(raw_df.y.as_matrix().shape)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

# Split train, test sets, and also return label_map
def train_test_split(df, train_ratio = 0.2, test_ratio = 0.1):
    
    test_x = []
    test_y = []
    train_x = []
    train_y = []
    for i in set(df.y.tolist()):
        tmp_df = df[df.y == i]
        tmp_df = shuffle(tmp_df)
        tmp_n = int(len(tmp_df)*train_ratio)
        tmp_m = int(len(tmp_df)*test_ratio)
        train_x += tmp_df.x.tolist()[: tmp_n]
        test_x += tmp_df.x.tolist()[tmp_n: tmp_n + tmp_m]
        train_y += tmp_df.y.tolist()[: tmp_n]
        test_y += tmp_df.y.tolist()[tmp_n: tmp_n + tmp_m]
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
## Parsing the data Frame into train and test sets
print("SPLITTING DATA INTO TRAIN AND TEST SETS!")
tr_x, tr_y, ts_x, ts_y = train_test_split(raw_df, 0.3, 0.1)
print(tr_x.shape)
print(tr_y.shape)
print(ts_x.shape)
print(ts_y.shape)
del raw_df
base = min(tr_x.min(), ts_x.min())
print(base)
tr_x = tr_x - base
ts_x = ts_x - base
UPPER_X = max(tr_x.max(), ts_x.max()) + 1
print(UPPER_X)
print(tr_x[0])
print(tr_x[0])
print(tr_x.max())
print(ts_x.max())
print(tr_x.min())
print(ts_x.min())

def comp_cls_wts(y, pwr = 0.5):
    dic = {}
    for x in set(y):
        dic[x] = len(y)**pwr/list(y).count(x)**pwr
    return dic
cls_wts = comp_cls_wts(tr_y)
print(cls_wts)
NUM_CLS = len(target_labels)
tr_y = np_utils.to_categorical(tr_y, num_classes=NUM_CLS)
ts_y = np_utils.to_categorical(ts_y, num_classes=NUM_CLS)
model = Sequential()
model.add(Embedding(UPPER_X, 128, input_length=L))
model.add(SimpleRNN(512))
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLS, activation='softmax'))
model.summary()
optimizer = SGD()
metrics = ['accuracy']
loss = 'categorical_crossentropy'
model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
res = model.fit(tr_x[:1000], tr_y[:1000], batch_size = 64,epochs = 5, validation_data = (ts_x[:330], ts_y[:330]), class_weight = cls_wts)
import gc
gc.collect()