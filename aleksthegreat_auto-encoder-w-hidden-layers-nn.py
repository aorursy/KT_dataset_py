from __future__ import print_function

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



import seaborn as sns

import gc



import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

from sklearn.metrics import mean_squared_error, roc_auc_score

from scipy.stats import norm, rankdata



import keras

from keras import regularizers

from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,PReLU, LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Conv2D, MaxPooling2D

from keras.optimizers import SGD

from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import tensorflow as tf
# reduce memory

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
train = reduce_mem_usage(pd.read_csv('../input/train.csv'))

test = reduce_mem_usage(pd.read_csv('../input/test.csv'))
features = [f for f in train if f not in ['ID_code','target']]
df_original = pd.concat([train, test],axis=0,sort=False)

df = df_original[features]

target = df_original['target'].values

id = df_original['ID_code']
from scipy.special import erfinv

trafo_columns = [c for c in df.columns if len(df[c].unique()) != 2]

for col in trafo_columns:

    values = sorted(set(df[col]))

    # Because erfinv(1) is inf, we shrink the range into (-0.9, 0.9)

    f = pd.Series(np.linspace(-0.9, 0.9, len(values)), index=values)

    f = np.sqrt(2) * erfinv(f)

    f -= f.mean()

    df[col] = df[col].map(f)
# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505

def auc_roc(y_true, y_pred):

    # any tensorflow metric

    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)



    # find all variables created for this metric

    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]



    # Add metric variables to GLOBAL_VARIABLES collection.

    # They will be initialized for new session.

    for v in metric_vars:

        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)



    # force to update metric values

    with tf.control_dependencies([update_op]):

        value = tf.identity(value)

        return value
from keras.callbacks import LearningRateScheduler

import math

from math import exp

from math import ceil



def step_decay(epoch):

    initial_lrate = 0.1

    drop = 0.5

    epochs_drop = 5.0

    lrate = initial_lrate * math.pow(drop,  

           math.floor((1+epoch)/epochs_drop))

    return lrate

        

def exp_decay(epoch):

    initial_lrate = 0.1

    k = 0.1

    t = epoch

    lrate = initial_lrate * exp(-k*t)

    return lrate





class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.lr = []

 

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

#        self.lr.append(exp_decay(len(self.losses)))

        self.lr.append(step_decay(len(self.losses)))
lrate = LearningRateScheduler(step_decay)

#lrate = LearningRateScheduler(exp_decay)

ao = ModelCheckpoint(filepath="auto_0.h5",save_best_only=True,verbose=0)

nn = ModelCheckpoint(filepath="nn_0.h5",save_best_only=True,verbose=0)

tb = TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=True)

rl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

loss_history = LossHistory()
from keras import backend as K

from keras.activations import elu

from keras.layers import Input, Dense, Lambda

from keras.models import Model

from keras.objectives import binary_crossentropy

from keras.callbacks import LearningRateScheduler

from keras import backend as K

from imblearn.keras import balanced_batch_generator

from imblearn.under_sampling import NearMiss, RandomUnderSampler, CondensedNearestNeighbour, AllKNN, InstanceHardnessThreshold

from sklearn.model_selection import KFold

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE

from keras.utils import multi_gpu_model

import math



verbose = 10

learning_rate = 0.0003

nb_epoch = int(3)

dcy = learning_rate / nb_epoch

batch_size = 256

encoding_dim =400

hidden_dim = int(encoding_dim*2) #i.e. 7

predictions = np.zeros(len(df))

label_cols = ["target"]

opt = keras.optimizers.SGD(lr=learning_rate, decay=dcy, nesterov=False)



def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(series.shape[1]))



trn_data, val_data = train_test_split(df[trafo_columns], test_size=0.3)

noisy_trn_data = add_noise(trn_data, 0.07)

input_dim = noisy_trn_data.shape[1] #num of columns



with tf.device('/cpu:0'):

    input_layer = Input(shape=(input_dim, ))



    x = Dense(hidden_dim, activation="relu", name="first", init='identity')(input_layer)

    x = Dense(hidden_dim, activation="relu", name='second')(x)

    x = BatchNormalization()(x)

    x = Dense(hidden_dim, activation="relu", name='third')(x)



    output_layer = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)    

    autoencoder.summary()

    

autoencoder.compile(metrics=['accuracy'],

                    loss='mean_squared_error',

                    optimizer=opt)
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())



    history = autoencoder.fit(noisy_trn_data, trn_data,

                        epochs=nb_epoch,

                        batch_size=batch_size,

                        shuffle=True,

                        validation_data=(val_data, val_data),

                        verbose=1,

                        callbacks=[ao,tb,es,loss_history,lrate])

# we build a new model with the activations of the old model

# this model is truncated after the first layer



#second_hidden_layer = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('second').output)

third_hidden_layer = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('third').output)



#print(second_hidden_layer.summary())

print(third_hidden_layer.summary())

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

#    get_2nd_hidden_layer = second_hidden_layer.predict(df)

    get_3rd_hidden_layer = third_hidden_layer.predict(df)

    

#print(get_2nd_hidden_layer.shape)

print(get_3rd_hidden_layer.shape)
del df_original, df, noisy_trn_data, test, train, trn_data, val_data

gc.collect()
#layer_output_2 = reduce_mem_usage(pd.DataFrame(get_2nd_hidden_layer))

layer_output_3 = reduce_mem_usage(pd.DataFrame(get_3rd_hidden_layer))
#hidden = np.concatenate([layer_output_2, layer_output_3], axis=1)

#hidden = pd.DataFrame(hidden)

hidden = pd.DataFrame(layer_output_3)

print(hidden.shape)



#del layer_output_2

del layer_output_3

gc.collect
#hidden

hidden['target'] = target

hidden['ID_code'] = id.values

print(hidden.head(5))
#find kernal here https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82863

features = [f for f in hidden if f not in ['ID_code','target']]

train = hidden[hidden['target'].notnull()]

test = hidden[hidden['target'].isnull()]

test = np.reshape(test[features].values, (-1,test[features].shape[1], 1))



len_input_columns, len_data = train.shape[1], train.shape[0]



print(train.shape)

print(test.shape)

print(train.head(5))
predictions = np.zeros(shape=(len(test), 1))

label_cols = ["target"]

train_x, valid_x, train_y, valid_y = train_test_split(train[features], train['target'], test_size=0.3)
with tf.device('/cpu:0'):



    input_dim = train_x.shape[1] #num of columns, 4500

    input_layer = Input(shape=(input_dim, 1))



    x = Dense(hidden_dim, activation='relu')(input_layer)

    x = Flatten()(x)





    output_layer = Dense(1, activation='sigmoid')(x)

    model= Model(inputs=input_layer, outputs=output_layer)

    model.summary()



opt = keras.optimizers.SGD(lr=learning_rate, decay=dcy, nesterov=False)

model.compile(metrics=['accuracy', auc_roc],

                    loss='binary_crossentropy',

                    optimizer=opt)
pos = (pd.Series(train_y == 1))



# Add positive examples

train_x_x = pd.concat([train_x, train_x.loc[pos]], axis=0)

train_y_y = pd.concat([train_y, train_y.loc[pos]], axis=0)



# Shuffle data

idx = np.arange(len(train_x_x))

np.random.shuffle(idx)

train_x_x = train_x_x.iloc[idx]

train_y_y = train_y_y.iloc[idx]



train_x_x = np.reshape(train_x.values, (-1, train_x.shape[1], 1))

valid_x_x = np.reshape(valid_x.values, (-1, valid_x.shape[1], 1))
history = model.fit(train_x_x, train_y,

                    epochs=1,

                    batch_size=int(128),

                    shuffle=True,

                    validation_data=(valid_x_x, valid_y),

                    verbose=1,

                    callbacks=[nn,tb,es,loss_history,lrate])



predictions = model.predict(test)