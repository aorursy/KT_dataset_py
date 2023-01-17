# Read basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
# Read the training data

data = pd.read_csv('../input/drugsComTrain_raw.csv')
# Show the head of the data

data.head()
# Create labels based on the original article: Grässer et al. (2018)

r = data['rating']

labels = -1*(r <= 4) + 1*(r >= 7)

# Add the label column to the data

data['label'] = labels

# Check the new data

data.head()
# Check ratings to labels conversion

import matplotlib.pyplot as plt

data.plot(x = 'rating', y = 'label', kind = 'scatter')

plt.show()
# Plot distribution of labels

data.hist(column = 'label', bins = np.arange(-1, 3), align = 'left');
data['review_length'] = data['review'].apply(len)

data['review_length'].describe()
data.hist('review_length', bins = np.arange(0, 1500, 100));

plt.title('Distribution of the reviews lengths')

plt.xlabel('Review length')

plt.ylabel('Count')

plt.show()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# Common settings for all models

WORDS = 1000

LENGTH = 100

N = 10000

DEPTH = 32



# Read a part of the reviews and create training sequences (x_train)

samples = data['review'].iloc[:N]

tokenizer = Tokenizer(num_words = WORDS)

tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

x_train = pad_sequences(sequences, maxlen = LENGTH)
from keras.utils import to_categorical



# Convert the labels to one_hot_category values

one_hot_labels = to_categorical(labels[:N], num_classes = 3)
# Check the training and label sets

x_train.shape, one_hot_labels.shape
# We use the same plotting commands several times, so create a function for that purpose

def plot_history(history):

    

    f, ax = plt.subplots(1, 2, figsize = (16, 7))

    

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1, len(acc) + 1)



    plt.sca(ax[0])

    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.sca(ax[1])

    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



    plt.show()
# Similarly create a function for model training, for demonstration purposes we use constant values

def train_model(model, x, y, e = 10, bs = 32, v = 1, vs = 0.25):

    h = model.fit(x, y, epochs = e, batch_size = bs, verbose = v, validation_split = vs)

    return h
from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense



# First model: Embedding layer -> Flatten -> Dense classifier

m0 = Sequential()

m0.add(Embedding(WORDS, DEPTH, input_length = LENGTH)) 

m0.add(Flatten())

m0.add(Dense(32, activation = 'relu'))

m0.add(Dense(3, activation = 'softmax'))

m0.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

m0.summary()
# Train the first model and plot the history

h0 = train_model(m0, x_train, one_hot_labels)

plot_history(h0)
from keras.layers import LSTM



# Second model: Embedding -> LSTM -> Dense classifier

m1 = Sequential()

m1.add(Embedding(WORDS, DEPTH, input_length = LENGTH))

m1.add(LSTM(DEPTH))

m1.add(Dense(3, activation = 'softmax'))

m1.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

m1.summary()
# Train the second model and plot the history

h1 = train_model(m1, x_train, one_hot_labels)

plot_history(h1)
from keras.layers import GRU



# Third model: Embedding -> GRU -> Dense classifier

m2 = Sequential()

m2.add(Embedding(WORDS, DEPTH, input_length = LENGTH))

m2.add(GRU(LENGTH))

m2.add(Dense(3, activation = 'softmax'))

m2.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

m2.summary()
# Train the third model and plot the history

h2 = train_model(m2, x_train, one_hot_labels)

plot_history(h2)
# Fourth model: Embedding -> GRU with dropouts -> Dense classifier

m3 = Sequential()

m3.add(Embedding(WORDS, DEPTH, input_length = LENGTH))

m3.add(GRU(DEPTH, dropout = 0.2, recurrent_dropout = 0.2))

m3.add(Dense(3, activation = 'softmax'))

m3.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

m3.summary()
# Train and plot the history

h3 = train_model(m3, x_train, one_hot_labels)

plot_history(h3)
# Fifth model: Embedding -> Stack of GRU layers -> Dense classifier

m4 = Sequential()

m4.add(Embedding(WORDS, DEPTH, input_length = LENGTH))

m4.add(GRU(DEPTH, dropout = 0.1, recurrent_dropout = 0.5, return_sequences = True))

m4.add(GRU(DEPTH, activation = 'relu', dropout = 0.1, recurrent_dropout = 0.5))

m4.add(Dense(3, activation = 'softmax'))

m4.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

m4.summary()
# Train and plot the history

h4 = train_model(m4, x_train, one_hot_labels)

plot_history(h4)
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D



# Sixth model: Embedding -> Conv1D & MaxPooling1D -> Dense classifier

m5 = Sequential()

m5.add(Embedding(WORDS, DEPTH, input_length = LENGTH))

m5.add(Conv1D(DEPTH, 7, activation = 'relu'))

m5.add(MaxPooling1D(5))

m5.add(Conv1D(DEPTH, 7, activation = 'relu'))

m5.add(GlobalMaxPooling1D())

m5.add(Dense(3, activation = 'softmax'))

m5.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

m5.summary()
# Train and plot the history

h5 = train_model(m5, x_train, one_hot_labels)

plot_history(h5)
# Seventh model: Embedding -> 2 x Conv1D + MaxPooling -> GRU -> Dense

m6 = Sequential()

m6.add(Embedding(WORDS, DEPTH, input_length = LENGTH))

m6.add(Conv1D(DEPTH, 5, activation = 'relu'))

m6.add(MaxPooling1D(5))

m6.add(Conv1D(DEPTH, 7, activation = 'relu'))

m6.add(GRU(DEPTH, dropout = 0.1, recurrent_dropout = 0.5))

m6.add(Dense(3, activation = 'softmax'))

m6.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

m6.summary()
# Train and plot the history

h6 = train_model(m6, x_train, one_hot_labels)

plot_history(h6)