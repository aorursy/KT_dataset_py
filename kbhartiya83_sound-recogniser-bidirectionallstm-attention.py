import numpy as np

import pandas as pd 

from scipy.io import wavfile

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

np.random.seed(101)

import IPython.display as ipd

import os

from scipy import signal

print(os.listdir("../input"))

from tqdm import tqdm, tqdm_notebook; tqdm.pandas() # Progress bar

from sklearn.metrics import label_ranking_average_precision_score

from sklearn.model_selection import train_test_split



# Machine Learning

import tensorflow as tf

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import (Dense, Bidirectional, CuDNNLSTM,

                          Dropout, LeakyReLU, Convolution2D, 

                          Conv2D, Conv1D)

from keras.models import Sequential

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping



import warnings; warnings.filterwarnings("ignore")

#print(os.listdir("../input/train_curated"))
train_cur = pd.read_csv("../input/train_curated.csv")

train_nos = pd.read_csv("../input/train_noisy.csv")

test = pd.read_csv("../input/sample_submission.csv")
train_cur.head()
train_nos.head()
test.head()
train_audio_cur = "../input/train_curated/"

ipd.Audio(train_audio_cur+"31a0f9cc.wav")

train_cur.loc[train_cur['fname']=='31a0f9cc.wav']
sampling_rate, data = wavfile.read(train_audio_cur+"31a0f9cc.wav")

print("Sampling Rate: {}".format(sampling_rate))

print("Data of the audio wave: {}".format(data))

print("Duration of Audio file: {}".format(len(data)/sampling_rate))
def plot_raw_wave(data):

    plt.figure(figsize=(16,9))

    plt.title("Raw Representation of Audio Wave")

    plt.ylabel("Amplitude")

    plt.plot(data)

plot_raw_wave(data)
#https://www.kaggle.com/davids1992/audio-representation-what-it-s-all-about

def log_specgram(audio, sample_rate, window_size=100,

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

    

    fig = plt.figure(figsize=(16,9))

    freqs, times, spectrogram = log_specgram(audio, sample_rate)

    plt.imshow(spectrogram.T, aspect='auto', origin='lower', 

               extent=[times.min(), times.max(), freqs.min(), freqs.max()])

    plt.yticks(freqs[::16])

    plt.xticks(times[::16])

    plt.title('Spectrogram')

    plt.ylabel('Freqs in Hz')

    plt.xlabel('Seconds')

    plt.show()

plot_log_specgram(data, sampling_rate, window_size=1000,step_size=10)
import librosa

import librosa.display
#https://www.kaggle.com/davids1992/audio-representation-what-it-s-all-about

S = librosa.feature.melspectrogram(data.astype(float), sr=sampling_rate, n_mels=128)



# Convert to log scale (dB). We'll use the peak power (max) as reference.

log_S = librosa.power_to_db(S, ref=np.max)



plt.figure(figsize=(16, 9))

librosa.display.specshow(log_S, sr=sampling_rate, x_axis='time', y_axis='mel')

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
def calculate_overall_lwlrap_sklearn(truth, scores):

    """Calculate the overall lwlrap using sklearn.metrics.lrap."""

    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.

    sample_weight = np.sum(truth > 0, axis=1)

    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

    overall_lwlrap = label_ranking_average_precision_score(

        truth[nonzero_weight_sample_indices, :] > 0, 

        scores[nonzero_weight_sample_indices, :], 

        sample_weight=sample_weight[nonzero_weight_sample_indices])

    return overall_lwlrap

def split_and_label(rows_label):

    labeled_rows = []

    for row in rows_label:

        rows_label = row.split(",")

        arr = np.zeros((80))

        for label in rows_label:

            idx = label_mapping[label]

            arr[idx] = 1

        labeled_rows.append(arr)

    return labeled_rows 
label_columns = test.columns[1:]

label_mapping = dict((label,index) for index, label in enumerate(label_columns))

print("Total Number of Classes: {}".format(len(label_columns)))
train_cur_labels = split_and_label(train_cur['labels'])

train_nos_labels = split_and_label(train_nos['labels'])
for col in label_columns:

    train_cur[col] = 0

    train_nos[col] = 0

train_cur[label_columns] = train_cur_labels

train_nos[label_columns] = train_nos_labels
train_cur.head()
train_nos.head()
input_shape = (890,128)

n_classes = 80

n_epochs = 500

hop_length = 347

fmin = 20

fmax = sampling_rate // 2

n_mels = 128

n_fft = n_mels*20

opt = Adam(0.003, beta_1=0.75, beta_2=0.85, amsgrad=True)

sampling_rate = 44100

duration = 7

samples = sampling_rate*duration

def read_audio(pathname, trim_long_data):

    y, sr = librosa.load(pathname, sr=sampling_rate)

    # trim silence

    if 0 < len(y): # workaround: 0 length causes error

        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    # make it unified length to conf.samples

    if len(y) > samples: # long enough

        if trim_long_data:

            y = y[0:0+samples]

    else: # pad blank

        padding = samples - len(y)    # add padding at both ends

        offset = padding // 2

        y = np.pad(y, (offset, samples - len(y) - offset), 'constant')

    return y



def audio_to_melspectrogram(audio):

    spectrogram = librosa.feature.melspectrogram(audio, 

                                                 sr=sampling_rate,

                                                 n_mels=n_mels,

                                                 hop_length=hop_length,

                                                 n_fft=n_fft,

                                                 fmin=fmin,

                                                 fmax=fmax)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    return spectrogram



def read_as_melspectrogram(pathname, trim_long_data, debug_display=False):

    x = read_audio(pathname, trim_long_data)

    mels = audio_to_melspectrogram(x)

    if debug_display:

        IPython.display.display(IPython.display.Audio(x, rate=sampling_rate))

        show_melspectrogram(mels)

    return mels



def convert_wav_to_image(df, source):

    X = []

    for i, row in tqdm_notebook(df.iterrows()):

        try:

            x = read_as_melspectrogram(f'{source[0]}/{str(row.fname)}', trim_long_data=True)

        except:

            x = read_as_melspectrogram(f'{source[1]}/{str(row.fname)}', trim_long_data=True)



        #x_color = mono_to_color(x)

        X.append(x.transpose())

        #df.loc[i, 'length'] = x.shape[1]

    return X
train = pd.concat([train_cur[:4000],train_nos[:2000]])

train_curated_path = '../input/train_curated/'

train_noisy_path = '../input/train_noisy/'

test_path = '../input/test/'

train.shape
%%time

#X = np.array(convert_wav_to_image(train_cur,source=[train_curated_path]))

#X = np.array(convert_wav_to_image(train_nos, source=[train_noisy_path]))

X = np.array(convert_wav_to_image(train, source=[train_curated_path, train_noisy_path]))
Y = train[label_columns].values
#

class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None

        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:

            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:

            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
model = Sequential()

model.add(Bidirectional(CuDNNLSTM(128,return_sequences=True),input_shape=input_shape))

model.add(Bidirectional(CuDNNLSTM(128,return_sequences=True),input_shape=input_shape))

model.add(Bidirectional(CuDNNLSTM(128,return_sequences=True),input_shape=input_shape))

model.add(Attention(input_shape[0]))

model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(n_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])

model.summary()
es = EarlyStopping(monitor='val_acc', mode='max', verbose=2, patience=10)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=101)

history = model.fit(np.array(x_train),

          y_train,

          batch_size=512,

          epochs=500,

          validation_data=(np.array(x_val), y_val),

          callbacks = [es]

                   )
y_train_pred = model.predict(np.array(x_train))

y_val_pred = model.predict(np.array(x_val))

train_lwlrap = calculate_overall_lwlrap_sklearn(y_train, y_train_pred)

val_lwlrap = calculate_overall_lwlrap_sklearn(y_val, y_val_pred)



# Check training and validation LWLRAP score

print('Training LWLRAP : {}'.format(round(train_lwlrap,4)))

print('Validation LWLRAP : {}'.format(round(val_lwlrap,4)))
X_test = np.array(convert_wav_to_image(test, [test_path]))

predictions = model.predict(np.array(X_test))

test[label_columns] = predictions

test.to_csv('submission.csv', index=False)