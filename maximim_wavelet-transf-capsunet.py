from __future__ import division
from __future__ import print_function

PREF = '016a05'
import os
from os import listdir
from os.path import isfile, join
import time
import warnings
import traceback
import gc
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
import scipy
import multiprocessing
import numba
import copy


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, GaussianNoise
from keras import callbacks
import keras.backend as K
import tensorflow as tf
from keras.layers import * 
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.models import load_model


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = './'  
DATA_DIR = '../input/train'  
INPUT_DATA_PATH = '../input/'

SCALE = 32
SIG_LEN = 4096
BATCH_SIZE = 20
NUM_THREADS = 2
SEG_LENGHT_HAT = 1
Y_SCALE = 16


# To reduce load
REDUCED = True
MAX_START_IDX = 300000000

random_state=21
if not os.path.exists('{}/{}__start_indices.npy'.format(OUTPUT_DIR,PREF)) or True:
    START_INDICES_SIZE = int(629e6 / 4096 / 4*10*1.25)
    limits = [5656574, 50085878, 104677356, 138772453, 187641820, 218652630, 245829585, 307838917, 338276287, 375377848, 419368880, 461811623, 495800225, 528777115, 585568144, 621985673]
    rnd_idxs = np.zeros(shape=(START_INDICES_SIZE,), dtype=np.int32)

    if REDUCED:
        max_start_idx = MAX_START_IDX - 150000
    else:
        max_start_idx = 629000000 # From train.csv shape[0]-150000
    
    np.random.seed(random_state)
    max_start_binned = int(max_start_idx/(4096*SEG_LENGHT_HAT))

    rnd_idxs = np.random.randint(0, max_start_binned, size=START_INDICES_SIZE, dtype=np.int32)
    rnd_idxs *= 4096 * SEG_LENGHT_HAT
    rnd_idxs = np.unique(rnd_idxs) # Borro los duplicados
    for limit in limits: # Borro los que estan cerca de la trnasicion.
        rnd_idxs = rnd_idxs[np.logical_not(np.logical_and(rnd_idxs>(limit-200000), rnd_idxs<(limit+200000)))]

    print('-'*64)
    print('Created:{}, from {} (limit: {}),  min:{}, and max:{}'.format(rnd_idxs.shape[0], START_INDICES_SIZE, max_start_idx, np.min(rnd_idxs), np.max(rnd_idxs) ))

    rnd_idxs = np.asarray(rnd_idxs)
    for _ in range(50):
        np.random.shuffle(rnd_idxs)
    print(rnd_idxs[:8])
    print(rnd_idxs[-8:])

    np.save('{}{}__start_indices.npy'.format(OUTPUT_DIR,PREF), rnd_idxs)
    print('{}{}__start_indices.npy saved!'.format(OUTPUT_DIR,PREF))
    print('-'*64)
    
    start_indices = rnd_idxs
else:
    start_indices = np.load('{}{}__start_indices.npy'.format(OUTPUT_DIR,PREF))


gtr_idxs = start_indices[:28000]
gts_idxs = start_indices[28000:40000]

if REDUCED:
    train_df = pd.read_csv(os.path.join(INPUT_DATA_PATH, 'train.csv'), 
                                dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, nrows=MAX_START_IDX)
else:
    train_df = pd.read_csv(os.path.join(INPUT_DATA_PATH, 'train.csv'), 
                                dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, nrows=MAX_START_IDX)

import pywt

def create_features(seg, X_hat):
    global SIG_LEN, SCALE
    scales = range(1,SCALE+1)
    waveletname = 'mexh'
    
    X = pywt.cwt(seg['acoustic_data'].values[2:-2], np.asarray(scales), waveletname)[0].reshape(1,SCALE,SIG_LEN-4)
    
    X -= np.min(X, axis=1)
    X /= np.max(X,axis=1)
    
    if X_hat is None:
        X_hat = X
    else:
        X_hat = np.concatenate([X_hat, X], axis=0)

    
    return X_hat
class DataGen(Sequence):

    def __init__(self, X=None, batch_size=12, PREF=PREF,OUTPUT_DIR=OUTPUT_DIR, DATA_DIR='../input/',
                 start_indices=[2,4,8], load_raw_data=True):
        self.start_indices=start_indices
        self.X = X
        self.PREF = PREF
        self.OUTPUT_DIR = OUTPUT_DIR
        self.DATA_DIR = DATA_DIR
        self.batch_size=batch_size
        self.load_raw_data = load_raw_data
        self.train_df = None
            
        if self.load_raw_data:
            print('Loading train.csv ...')
            self.train_df = pd.read_csv(os.path.join(self.DATA_DIR, 'train.csv'), 
                                dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}) #, nrows=100000)
        return

    def __len__(self):
        return int(np.floor(1.0 * len(self.start_indices)/self.batch_size))

    def get_raw_data(self):
        
        return self.train_df.copy()

    def set_raw_data(self, train_df):
        self.train_df = train_df
        return
    
    def __getitem__(self, idx):
        p_id, seg_st = 0, 0

        return self.process(p_id, idx,seg_st)
    

    def generate(self):
        idx = 0
        while True:

            if idx ==  self.__len__():
                idx = 0
            
            yield(self.__getitem__(idx))
            idx += 1

        return
    
    def process(self, p_id=0, idx=[0], seg_st=0): # idx = batched pos.
        try:
            start_indices = self.start_indices[int(self.batch_size * idx):int(self.batch_size * (idx+1))]
            seg_st += self.batch_size * idx

            X1, X2 = None, None
            train_y = pd.DataFrame(dtype=np.float64, columns=['time_to_failure'])

            for seg_id, start_idx in zip(range(int(seg_st), int(seg_st + self.batch_size)), start_indices):
                end_idx = np.int32(start_idx + SIG_LEN)
                seg = self.train_df.iloc[start_idx: end_idx,:].copy()
                X1 = create_features(seg, X1)
                train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1] / Y_SCALE

            return X1.reshape(-1,X1.shape[1],X1.shape[2],1), train_y
        except:
            print('Error working with: %d, %d, %d to %d of 629e6' % (p_id, 0, start_idx, end_idx))
            print(traceback.format_exc())
            success = 0
            print('Error working with: %d, %d, %d to %d of 629e6' % (p_id, 0, start_idx, end_idx))
        
        return

Gtr = DataGen(start_indices=gtr_idxs, batch_size=BATCH_SIZE, load_raw_data=False ) 
Gtr.set_raw_data(train_df)

Gval = DataGen(start_indices=gts_idxs, batch_size=BATCH_SIZE, load_raw_data=False ) 
Gval.set_raw_data(train_df)

# Capsule definition
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(Layer):
    def __init__(self,**kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = 12
        self.dim_capsule = 16
        self.routings = 3
        self.share_weights = True
        self.activation = squash


    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
# MODEL DEF
def _Model(shape):
    units=16
    guess_dim=True
    if guess_dim == True:
        n = SCALE
        dma = 50000
    inp = Input(shape=(n,dma,))

    main_input = Input(shape=(shape[1], shape[2], shape[3]), name='main_input') 
    x = Conv2D(72, (3, 3), activation='relu', name='Conv_0')(main_input)
    x = Conv2D(32, (3, 3), activation='relu', name='Conv_1')(x)
    x = Reshape((-1, 128))(x)
    
    x = Capsule()(x)
    x = Flatten()(x)

    x = Dense(units=128)(x)
    x = Dense(units=128)(x)
    x = Dropout(0.2)(x)    
    # The output layer
    preds = Dense(units=1, activation='linear')(x)
    model = Model(inputs=main_input, outputs=preds)

    model.summary()

    # Compile and fit model
    model.compile(optimizer=Adam(0.001),loss='mae', metrics=['mean_squared_error']) 
    
    return model

model = _Model([BATCH_SIZE,SCALE,SIG_LEN-4,1])
print('training ...')
model.fit_generator(Gtr, epochs=1, verbose=1, steps_per_epoch=2, #len(Gtr), 
            validation_steps=None,
            max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

