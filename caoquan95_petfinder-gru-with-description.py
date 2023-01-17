import numpy as np

import pandas as pd

import re

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

import os

from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense, GRU

import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
data = train_df[['AdoptionSpeed', "Description"]]

data = data[data['Description'] != '']

descriptions = data['Description'].apply(lambda x: re.sub(r"[^a-z0-9 ]+", "", str(x).lower()))

labels = data['AdoptionSpeed']
labels_oh = to_categorical(labels)
tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(descriptions)

word_index = tokenizer.word_index



print('Found %s unique tokens' % len(word_index))

sequences = tokenizer.texts_to_sequences(descriptions)
max_len = 100

max_words = 10000

training_sample = 10000

val_sample = 12000

padded_sequences = pad_sequences(sequences, maxlen=max_len)

print("Shape of data: ", padded_sequences.shape)

print("Shape of labels: ", labels_oh.shape)
# Shuffle the data

indices = np.arange(data.shape[0])
np.random.shuffle(indices)

padded_sequences = padded_sequences[indices]

labels = labels[indices]



x_train = padded_sequences[:training_sample]

y_train = labels_oh[:training_sample]

x_val = padded_sequences[training_sample: val_sample]

y_val = labels_oh[training_sample: val_sample]

x_test = padded_sequences[val_sample: ]

y_test = labels_oh[val_sample: ]
# load Glove word embedding

embeddings_index = {}

f = open("../input/glove6b300dtxt/glove.6B.300d.txt")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 300

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():

    if i < max_words:

        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector
model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=max_len))

model.add(GRU(64, return_sequences=True))

model.add(GRU(64))

model.add(Dense(5, activation="softmax"))

model.summary()
model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train,

    epochs=10,

    batch_size=256,

    validation_data=(x_val, y_val))

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
x_train.shape