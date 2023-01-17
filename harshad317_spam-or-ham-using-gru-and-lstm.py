import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf
df = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
df.head(10)
import string

import re
def preprocess_text(sen):

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower()

df['Message'] = df['Message'].apply(preprocess_text)
df.head()
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
msg = df['Message']
vocab_size = 10000

embedding_dim = 16

max_length = 32

trunc_type = 'post'

pad_type = 'post'

oov_tok = '<OOV>'
token = Tokenizer(num_words= vocab_size, oov_token= oov_tok)

token.fit_on_texts(df['Message'])

word_index = token.word_index

word_index
sens = token.texts_to_sequences(df['Message'])

padded = pad_sequences(sequences= sens, maxlen= max_length, padding= 'post', truncating= trunc_type)
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(padded, df['Category'], test_size = 0.2)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# Model Definition with LSTM

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),

    tf.keras.layers.Dense(6, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs= 5, validation_data= (x_test, y_test))
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()



plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
# Model Definition with LSTM

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(6, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs= 5, validation_data= (x_test, y_test))
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()



plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')