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
from numpy import array

from numpy import asarray

from numpy import zeros

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Embedding

# define documents

docs = ['Well done!',

		'Good work',

		'Great effort',

		'nice work',

		'Excellent!',

		'Weak',

		'Poor effort!',

		'not good',

		'poor work',

		'Could have done better.']

# define class labels

labels = array([1,1,1,1,1,0,0,0,0,0])

# prepare tokenizer

t = Tokenizer()

t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1

# integer encode the documents

encoded_docs = t.texts_to_sequences(docs)

print(encoded_docs)

# pad documents to a max length of 4 words

max_length = 4

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs)

# load the whole embedding into memory

embeddings_index = dict()

f = open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')

for line in f:

	values = line.split()

	word = values[0]

	coefs = asarray(values[1:], dtype='float32')

	embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs

embedding_matrix = zeros((vocab_size, 100))

for word, i in t.word_index.items():

	embedding_vector = embeddings_index.get(word)

	if embedding_vector is not None:

		embedding_matrix[i] = embedding_vector

# define model

model = Sequential()

e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)

model.add(e)

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model

print(model.summary())

# fit the model

model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print('Accuracy: %f' % (accuracy*100))