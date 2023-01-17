# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# collect reviews into a list of strings

imdb_dir  = '/kaggle/input/imdb-data-raw-text/aclimdb/aclImdb/'

train_dir = os.path.join(imdb_dir, 'train')

labels = []

texts = []

for label_type in ['neg', 'pos']:

    dir_name = os.path.join(train_dir, label_type)

    for fname in os.listdir(dir_name):

        if fname[-4:] == '.txt':

            f = open(os.path.join(dir_name, fname))

            texts.append(f.read())

            f.close()

            if label_type == 'neg':

                labels.append(0)

            else:

                labels.append(1)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import numpy as np

maxlen = 100  # cuts off review after 100 words

training_samples = 200  # train on 200 samples

validation_samples = 10000  # validate on 10000 samples

max_words = 10000  # consider only the top 10000 words 

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



len(sequences), len(sequences[4])
data = pad_sequences(sequences, maxlen=maxlen)

data.shape
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)
# shuffle the training/validation set (original set is ordered)

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]



x_train = data[:training_samples]

y_train = labels[:training_samples]

x_val = data[training_samples: training_samples + validation_samples]

y_val = labels[training_samples: training_samples + validation_samples]
glove_dir = '/kaggle/input/glove-global-vectors-for-word-representation/'

embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

i = 0

for line in f:

    print(line)

    i+=1

    if i == 3:

        break

# /kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt

# /kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt

# /kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt

# parsing the GloVe word-embeddings file

glove_dir = '/kaggle/input/glove-global-vectors-for-word-representation/'

embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Preparing the GloVe word-embeddings matrix

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():  # work_index is a dictionary

    if i < max_words:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            # Words not found in the embedding index will be all zeros. 

            embedding_matrix[i] = embedding_vector
from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense

model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=maxlen))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()
# load the Glove embedding in the model

model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = False  # freeze the Embedding layer
model.compile(optimizer = 'rmsprop',

             loss = 'binary_crossentropy',

             metrics = ['acc'])

history = model.fit(x_train, y_train, 

                   epochs = 10,

                   batch_size=32,

                    validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')
import matplotlib.pyplot as plt

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
from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense

model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=maxlen))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',

                loss='binary_crossentropy',

                metrics=['acc'])

history = model.fit(x_train, y_train,

                    epochs=10,

                    batch_size=32,

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
test_dir = os.path.join(imdb_dir, 'test')

labels = []

texts = []

for label_type in ['neg', 'pos']:

    dir_name = os.path.join(test_dir, label_type)

    for fname in sorted(os.listdir(dir_name)):

        if fname[-4:] == '.txt':

            f = open(os.path.join(dir_name, fname))

            texts.append(f.read())

            f.close()

            if label_type == 'neg':

                labels.append(0)

            else:

                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)

x_test = pad_sequences(sequences, maxlen=maxlen)

y_test = np.asarray(labels)
model.load_weights('pre_trained_glove_model.h5')

model.evaluate(x_test, y_test)