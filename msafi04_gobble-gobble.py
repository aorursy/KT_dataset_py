# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, auc
from IPython.display import display
import gc

import tensorflow as tf

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, Bidirectional, BatchNormalization, Flatten
from keras.layers import Dropout, Activation, GlobalMaxPool1D
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import initializers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import keras.backend as K

import random
seed = 42
random.seed(42)

import os
print(os.listdir("../input"))

from time import time, strftime, gmtime
start = time()
train = pd.read_json('../input/train.json')
print(train.shape)
display(train.head())
test = pd.read_json('../input/test.json')
print(test.shape)
display(test.head())
#target = to_categorical(train['is_turkey'].values)
target = np.asarray(train['is_turkey'].values)
print(target.shape)

maxlen = 10

train_pad = pad_sequences(train['audio_embedding'].tolist(), maxlen = 10)
test_pad = pad_sequences(test['audio_embedding'].tolist(), maxlen = 10)

Xtrain, Xvalid, ytrain, yvalid = train_test_split(train_pad, target, 
                                                  test_size = 0.2)
print(Xtrain.shape, ytrain.shape, Xvalid.shape, yvalid.shape)
def as_keras_metric(method):
    import functools
    
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
auc_roc = as_keras_metric(tf.metrics.auc)
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate, Reshape, Flatten, Concatenate

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
inputs = Input(shape = (10, 128 ))
x = BatchNormalization()(inputs)
#x = Bidirectional(LSTM(128, activation = 'relu', return_sequences = True, 
 #                      dropout = 0.4, recurrent_dropout = 0.4))(x)
x = Bidirectional(LSTM(64, activation = 'relu', return_sequences = True, 
                       dropout = 0.4, recurrent_dropout = 0.4))(x)
x = Attention(10)(x)
x = Dense(32, activation = "relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation = "sigmoid")(x)
model = Model(inputs = inputs, outputs = x)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', auc_roc])

print(model.summary())
hist = model.fit(Xtrain, ytrain, batch_size = 128, epochs = 20, 
          validation_data = (Xvalid, yvalid), 
          verbose = 2
         )
vpred = model.predict(Xvalid, verbose = 1)
#Predict test set

tpred = model.predict(test_pad, batch_size = 256, verbose = 1)
submission = pd.DataFrame({'vid_id':test['vid_id'],'is_turkey':[x[0] for x in tpred]})
print(submission.shape)
display(submission.head())
submission.to_csv('./submission.csv', index = False)
finish = time()
print(strftime("%H:%M:%S", gmtime(finish - start)))
