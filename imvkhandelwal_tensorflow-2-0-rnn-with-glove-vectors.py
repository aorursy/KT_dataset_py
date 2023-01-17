import json

import tensorflow as tf

import csv

import random

import numpy as np



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import regularizers





embedding_dim = 100

max_length = 16

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"

training_size=160000

test_portion=.1



corpus = []
import pandas as pd

df = pd.read_csv('../input/glovedataset/training_cleaned.csv')

df.head()
num_sentences = 0



with open("../input/glovedataset/training_cleaned.csv") as csvfile:

    reader = csv.reader(csvfile, delimiter=',')

    for row in reader:

        list_item=[]

        list_item.append(row[5])

        this_label=row[0]

        if this_label=='0':

            list_item.append(0)

        else:

            list_item.append(1)

        num_sentences = num_sentences + 1

        corpus.append(list_item)
print(num_sentences)

print(len(corpus))

print(corpus[1])

sentences=[]

labels=[]

random.shuffle(corpus)

for x in range(training_size):

    sentences.append(corpus[x][0])

    labels.append(corpus[x][1])

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

vocab_size=len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[0:split]

training_sequences = padded[split:training_size]

test_labels = labels[0:split]

training_labels = labels[split:training_size]
print(vocab_size)

print(word_index['i'])
embeddings_index = {};

with open('../input/glovedataset/glove.6B.100d.txt') as f:

    for line in f:

        values = line.split();

        word = values[0];

        coefs = np.asarray(values[1:], dtype='float32');

        embeddings_index[word] = coefs;



embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word);

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector;
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(64, 5, activation='relu'),

    tf.keras.layers.MaxPooling1D(pool_size=4),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()



num_epochs = 50

history = model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)



print("Training Complete")

import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()