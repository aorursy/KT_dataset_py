import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt
file_path = '/kaggle/input/the-social-dilemma-tweets/TheSocialDilemma.csv'

data = pd.read_csv(file_path)

data = data[['text', 'Sentiment']]

data.head()



def clean_text(text):

  text = text.lower()

  text = re.sub('\[.*?\]', '', text)

  text = re.sub('https?://\S+|www\.\S+', '', text)

  text = re.sub('\n', '', text)

  text = " ".join(filter(lambda x:x[0]!="@", text.split()))

  return text

data['text'] = data['text'].apply(lambda x: clean_text(x))



X = data['text']

y = data['Sentiment'].map({'Negative':0, 'Neutral':1, 'Positive':2})



train_size = int(len(data)*0.8)

X_train, y_train = X[:train_size], y[:train_size]

X_test, y_test = X[train_size:], y[train_size:]

print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)

print("y_test shape:", y_test.shape)



print ('Length of text: {} characters'.format(len(X_train)))



print("Max tweet length:", X.map(len).max())

print("Min tweet length:", X.map(len).min())

print("Average tweet length:", X.map(len).mean())
vocab_size = 8000

embedding_dim = 32

max_length = 90

tokenizer = Tokenizer(vocab_size)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index



train_sequences = tokenizer.texts_to_sequences(X_train)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='pre', truncating='pre')

test_sequences = tokenizer.texts_to_sequences(X_test)

test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='pre', truncating='pre')



print("Shape of train_padded:", train_padded.shape)

print("Shape of test_padded:", test_padded.shape)
model = Sequential([

                    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

                    tf.keras.layers.LSTM(100),

                    tf.keras.layers.Dense(max_length/2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),

                    tf.keras.layers.Dropout(0.4),

                    tf.keras.layers.Dense(3, activation='softmax')

])

model.summary()



lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    0.01,

    decay_steps=10000,

    decay_rate=0.95,

    staircase=True

)



model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = lr_schedule),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(train_padded, y_train, epochs=30, validation_data=(test_padded, y_test))



def get_encode(x):

  #x = clean_text(x)

  x = tokenizer.texts_to_sequences(x)

  x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length, padding='pre', truncating='pre')

  return x



test_comment = ['This movie depicted the current society issues so well, I loved it so much']



seq = tokenizer.texts_to_sequences(test_comment)

padded = pad_sequences(seq, maxlen=max_length, padding='pre', truncating='pre')

print(padded.shape)

y_pred = model.predict(padded).round()

print(y_pred)