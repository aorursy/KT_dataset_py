import re

from sklearn.utils import shuffle

import numpy as np

import bz2

import tensorflow as tf

from tensorflow.keras.layers import *

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import pandas as pd
###load from check point

max_features = 8192

maxlen = 128

embed_size = 100

x_train = np.load('../input/preprocess/x_train.npy')

x_test = np.load('../input/preprocess/x_test.npy')

y_train = np.load('../input/preprocess/y_train.npy')

y_test = np.load('../input/preprocess/y_test.npy')

embedding_matrix = np.load('../input/preprocess/embedding_matrix.npy')
x_train.shape
mixedModel = tf.keras.Sequential([

    Embedding(max_features, embed_size , input_shape=(maxlen,),weights=[embedding_matrix], trainable=True),

    Dropout(0.2),

    BatchNormalization(),

    Conv1D(256,7,padding='same',activation='relu'),

    BatchNormalization(),

    Conv1D(128,3,padding='same',activation='relu'),

    BatchNormalization(),

    Conv1D(64,3,padding='same',activation='relu'),

    BatchNormalization(),

    Conv1D(64,3,padding='same',activation='relu'),

    CuDNNLSTM(64,return_sequences=True),

    CuDNNLSTM(64,return_sequences=True),

    CuDNNLSTM(64,return_sequences=False),

    Dense(32,activation='relu'),

    Dense(2, activation='softmax')

])

mixedModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
mixed_his = mixedModel.fit(x_train, y_train, batch_size=6000, epochs=24, validation_split=0.1,verbose=1)
history_df = pd.DataFrame(data=mixed_his.history)

history_df.to_csv('mixed_his.csv')
test_acc = mixedModel.evaluate(x_test,y_test)

print(test_acc)
with open('test_acc.txt', 'w') as f:

    result_arr = [str(x) for x in test_acc]

    result_str = ",".join(result_arr)

    f.write(result_str)
# Save the model

mixedModel.save('mixedModel.h5')
convModel = tf.keras.Sequential([

    Embedding(max_features, embed_size , input_shape=(maxlen,),weights=[embedding_matrix], trainable=True),

    Dropout(0.2),

    BatchNormalization(),

    Conv1D(128,7,padding='same',activation='relu'),

    BatchNormalization(),

    Conv1D(128,3,padding='same',activation='relu'),

    BatchNormalization(),

    Conv1D(64,3,padding='same',activation='relu'),

    BatchNormalization(),

    Conv1D(64,3,padding='same',activation='relu'),

    BatchNormalization(),

    Conv1D(64,3,padding='same',activation='relu'),

    BatchNormalization(),

    Flatten(),

    Dense(32,activation='relu'),

    Dense(2, activation='softmax')

])

convModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
#conv_his = convModel.fit(x_train, y_train, batch_size=10000, epochs=16, validation_split=0.1,verbose=1)
#history_df = pd.DataFrame(data=conv_his.history)

#history_df.to_csv('conv_his.csv')
#convModel.evaluate(x_test,y_test)