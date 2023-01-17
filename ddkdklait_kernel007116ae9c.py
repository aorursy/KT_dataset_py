from keras.datasets import imdb

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM, Flatten, Dropout

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import os

from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from sklearn.metrics import classification_report

from tensorflow import set_random_seed

import random as rn



set_random_seed(1)

rn.seed(1)

np.random.seed(1)



num_words = 2000

old = np.load

np.load = lambda *a, **k: old(*a,allow_pickle=True,**k)



(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",

                                                      num_words=num_words,

                                                      skip_top=0,

                                                      maxlen=None,

                                                      seed=113,

                                                      start_char=1,

                                                      oov_char=2,

                                                      index_from=3)
max_review_length = 250

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vector_length = 32

model = Sequential()

model.add(Embedding(input_dim=num_words, output_dim=embedding_vector_length, input_length=max_review_length))

model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dense(units=256, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
with tf.device('/device:GPU:0'):

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),

                ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)]

    

    train_history = model.fit(X_train, y_train, batch_size=32,

                              epochs=1000, verbose=2,

                              validation_split=0.2,

                              callbacks=callbacks)
with tf.device('/device:GPU:0'):

    model.load_weights('weights.hdf5')

    predict=model.predict_classes(X_test)

    predict_classes=predict.reshape(len(X_test))

    print(classification_report(predict_classes,y_test))