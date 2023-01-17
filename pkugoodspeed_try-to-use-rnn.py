# input, output and command line tools

import os

from os.path import isdir, join

import pandas as pd



# math and data handler

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA



# audio file i/o

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display





mpl.rc('font', family = 'serif', size = 17)

mpl.rcParams['xtick.major.size'] = 5

mpl.rcParams['xtick.minor.size'] = 2

mpl.rcParams['ytick.major.size'] = 5

mpl.rcParams['ytick.minor.size'] = 2



# Shuffle data

from sklearn.utils import shuffle



# Keras

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

from keras.layers.recurrent import SimpleRNN, LSTM #Actually in this test, SimpleRNN works much better

from keras.layers.embeddings import Embedding
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



TAGET_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
## Function for loading the audio data, return a dataFrame

MAX_SIZE = 16000

def load_audio_data(path, ltoi):

    '''

    path: audio file path

    return: pd.DataFrame

    '''

    x = []

    y = []

    for i, folder in enumerate(os.listdir(path)):

        for filename in os.listdir(path + '/' + folder):

            if filename == 'README.md':

                continue

            rate, sample = wavfile.read(data_dir + '/' + folder + '/' + filename)

            assert(rate == MAX_SIZE)

            if folder == '_background_noise_':

                length = len(sample)

                for j in range(int(length/rate)):

                    x.append(np.array(sample[j*rate: (j+1)*rate]))

                    y.append(ltoi['silence'])

            else:

                x.append(np.array(sample))

                label = folder

                if folder not in TAGET_LABELS:

                    label = 'unknown'

                y.append(ltoi[label])

    x = np.array(pad_sequences(x, maxlen=MAX_SIZE))

    y = np.array(y)

    df = pd.DataFrame()

    df['x'] = list(x)

    df['y'] = list(y)

    return df
data_dir = '../input/train/audio'

os.listdir('{0}/_background_noise_'.format(data_dir))
## Loading raw data Frame

print("LOADING RAW DATA!")

label2idx = {}

idmap = {}

for i,lab in enumerate(TAGET_LABELS):

    label2idx[lab] = i

    idmap[i] = lab

raw_df = load_audio_data(data_dir, label2idx)

print(label2idx)

print(idmap)

print(raw_df.x.as_matrix().shape)

print(raw_df.y.as_matrix().shape)
# Split train, test sets, and also return label_map

def train_test_split(df, train_ratio = 0.2, test_ratio = 0.1):

    '''

    return train_sets + test_sets + label_map, which maps from y to label name

    '''

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
BASE_LVL = min(tr_x.min(), ts_x.min())

print(BASE_LVL)

tr_x = tr_x - BASE_LVL

ts_x = ts_x - BASE_LVL

UPPER_X = max(tr_x.max(), ts_x.max()) + 1

print(UPPER_X)

print(tr_x[0])
print(tr_x[0])

print(tr_x.max())

print(ts_x.max())

print(tr_x.min())

print(ts_x.min())
# Function to compute class weights

def comp_cls_wts(y, pwr = 0.5):

    '''

    Used to compute class weights

    '''

    dic = {}

    for x in set(y):

        dic[x] = len(y)**pwr/list(y).count(x)**pwr

    return dic
cls_wts = comp_cls_wts(tr_y)

print(cls_wts)
NUM_CLS = len(TAGET_LABELS)

tr_y = np_utils.to_categorical(tr_y, num_classes=NUM_CLS)

ts_y = np_utils.to_categorical(ts_y, num_classes=NUM_CLS)
model = Sequential()

model.add(Embedding(UPPER_X, 128, input_length=MAX_SIZE))

model.add(SimpleRNN(512))

model.add(Dense(64, activation='relu'))

model.add(Dense(NUM_CLS, activation='softmax'))

model.summary()
### Compile the model

optimizer = SGD()

metrics = ['accuracy']

loss = 'categorical_crossentropy'

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
res = model.fit(tr_x, tr_y, batch_size = 64,epochs = 15, validation_data = (ts_x, ts_y),

                class_weight = cls_wts)