# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json

datastore = []

for line in open('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json','r'):

    datastore.append(json.loads(line))
sentences = []

labels = []

for item in datastore:

    sentences.append(item['headline'])

    labels.append(item['is_sarcastic'])
print(sentences[0])

print(labels[0])
import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.text import  Tokenizer

from keras.preprocessing.sequence import pad_sequences

vocab_size = 10000

max_length = 50

embedding_dim = 200 

padd_type = 'post'

trunc_type = 'post'



tokenizer = Tokenizer(num_words = vocab_size)

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

print(len(word_index))
sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences,maxlen = max_length,padding = padd_type, truncating = trunc_type )
print(type(padded))
labels = np.array(labels)

print(type(labels))
from keras.layers import *
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(64, 5, activation='relu'),

    tf.keras.layers.MaxPooling1D(pool_size=4),

    tf.keras.layers.LSTM(128,return_sequences = True),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(24,activation = 'relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, labels,epochs = num_epochs, verbose = 1,validation_split = 0.2)
print(history.history['val_accuracy'][9])

print(history.history['val_loss'][9])
import matplotlib.pyplot as plt



fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

fig.suptitle("Performance of Model without pretrained embeddings")

ax1.plot(history.history['accuracy'])

ax1.plot(history.history['val_accuracy'])

vline_cut = np.where(history.history['val_accuracy'] == np.max(history.history['val_accuracy']))[0][0]

ax1.axvline(x=vline_cut, color='k', linestyle='--')

ax1.set_title("Model Accuracy")

ax1.legend(['train', 'test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

ax2.axvline(x=vline_cut, color='k', linestyle='--')

ax2.set_title("Model Loss")

ax2.legend(['train', 'test'])

plt.show()