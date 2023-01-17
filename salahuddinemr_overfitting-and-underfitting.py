# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install tensorflow==2.0.0-rc1

import tensorflow as tf

from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
imdb = keras.datasets.imdb



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()


# The first indices are reserved

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0

word_index["<START>"] = 1

word_index["<UNK>"] = 2  # unknown

word_index["<UNUSED>"] = 3



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])



train_data = keras.preprocessing.sequence.pad_sequences(train_data,

                                                        value=word_index["<PAD>"],

                                                        padding='post',

                                                        maxlen=256)



test_data = keras.preprocessing.sequence.pad_sequences(test_data,

                                                       value=word_index["<PAD>"],

                                                       padding='post',

                                                       maxlen=256)
print(train_data[0])
x_val = train_data[:10000]

partial_x_train = train_data[10000:]



y_val = train_labels[:10000]

partial_y_train = train_labels[10000:]
# input shape is the vocabulary count used for the movie reviews (10,000 words)

vocab_size = 10000



model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 128))

model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(128, activation=tf.nn.relu))

model.add(keras.layers.Dense(128, activation=tf.nn.relu))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



model.summary()



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc','binary_crossentropy'])



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=40,

                    batch_size=512,callbacks=[callback],

                    validation_data=(x_val, y_val),

                    verbose=2)
results = model.evaluate(test_data, test_labels)



print(results)
modelL2 = keras.Sequential()

modelL2.add(keras.layers.Embedding(vocab_size, 128))

modelL2.add(keras.layers.GlobalAveragePooling1D())

modelL2.add(keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))

modelL2.add(keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))

modelL2.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



modelL2.summary()



modelL2.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc','binary_crossentropy'])



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

L2_history= modelL2.fit(partial_x_train,

                    partial_y_train,

                    epochs=40,

                    batch_size=512,callbacks=[callback],

                    validation_data=(x_val, y_val),

                    verbose=2)
L2_result = modelL2.evaluate(test_data,test_labels)

print(L2_result)
modeldrop = keras.Sequential()

modeldrop.add(keras.layers.Embedding(vocab_size, 128))

modeldrop.add(keras.layers.GlobalAveragePooling1D())

modeldrop.add(keras.layers.Dense(128, activation=tf.nn.relu))

keras.layers.Dropout(0.5),

modeldrop.add(keras.layers.Dense(128, activation=tf.nn.relu))

keras.layers.Dropout(0.5),

modeldrop.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



modeldrop.summary()



modeldrop.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc','binary_crossentropy'])



callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history_drop = modeldrop.fit(partial_x_train,

                    partial_y_train,

                    epochs=40,

                    batch_size=512,callbacks=[callback],

                    validation_data=(x_val, y_val),

                    verbose=2)
drop_result = modeldrop.evaluate(test_data,test_labels)

print(drop_result)
import matplotlib.pyplot as plt

def plot_history(histories, key='binary_crossentropy'):

  plt.figure(figsize=(16,10))



  for name, history in histories:

    val = plt.plot(history.epoch, history.history['val_'+key],

                   '--', label=name.title()+' Val')

    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),

             label=name.title()+' Train')



  plt.xlabel('Epochs')

  plt.ylabel(key.replace('_',' ').title())

  plt.legend()



  plt.xlim([0,max(history.epoch)])
plot_history([('baseline', history),

              ('dropout', history_drop),('regularization',L2_history)])