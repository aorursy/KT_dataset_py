# https://deeplearningcourses.com/c/deep-learning-advanced-nlp

from __future__ import print_function, division

from builtins import range

# Note: you may need to update your version of future

# sudo pip install -U future





import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.models import Model

from keras.layers import Dense, Embedding, Input

from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score



import keras.backend as K

# if len(K.tensorflow_backend._get_available_gpus()) > 0:

#   from keras.layers import CuDNNLSTM as LSTM

#   from keras.layers import CuDNNGRU as GRU

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# some configuration

MAX_SEQUENCE_LENGTH = 100

MAX_VOCAB_SIZE = 20000

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2

BATCH_SIZE = 128

EPOCHS = 5

# load in pre-trained word vectors

print('Loading word vectors...')

word2vec = {}

with open("/kaggle/input/glove-vectors/glove.6B.100d.txt") as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

  for line in f:

    values = line.split()

    word = values[0]

    vec = np.asarray(values[1:], dtype='float32')

    word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))

# prepare text samples and their labels

print('Loading in comments...')



train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

sentences = train["comment_text"].fillna("DUMMY_VALUE").values

possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

targets = train[possible_labels].values

sentences[1]
targets[0]
# convert the sentences (strings) into integers

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))



# pad sequences so that we get a N x T matrix

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)
# prepare embedding matrix

print('Filling pre-trained embeddings...')

num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx.items():

  if i < MAX_VOCAB_SIZE:

    embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

      embedding_matrix[i] = embedding_vector



# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(

  num_words,

  EMBEDDING_DIM,

  weights=[embedding_matrix],

  input_length=MAX_SEQUENCE_LENGTH,

  trainable=False

)



# create an LSTM network with a single LSTM

input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))

x = embedding_layer(input_)

print(x.shape)

x = LSTM(15, return_sequences=True)(x)

print(x.shape)

# x = Bidirectional(LSTM(15, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

print(x.shape)

output = Dense(len(possible_labels), activation="sigmoid")(x)

print(output.shape)
model = Model(input_, output)

model.compile(

  loss='binary_crossentropy',

  optimizer=Adam(lr=0.01),

  metrics=['accuracy'],

)
print('Training model...')

r = model.fit(

  data,

  targets,

  batch_size=BATCH_SIZE,

  epochs=EPOCHS,

  validation_split=VALIDATION_SPLIT

)
test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")

test.head()
# prepare text samples and their labels

print('Loading test comments...')



train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")

sentences_test = train["comment_text"].fillna("DUMMY_VALUE").values

# convert the sentences (strings) into integers

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(sentences_test)

sentences_test = tokenizer.texts_to_sequences(sentences_test)

sentences_test
# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))

# pad sequences so that we get a N x T matrix

data = pad_sequences(sentences_test, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)
y_pred = model.predict(data)

y_pred
y_pred[0]
id = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')

id.head()
output = pd.DataFrame({'id': id.id, 'toxic': y_pred[:,0], 'severe_toxic': y_pred[:,1], 'obscene': y_pred[:,2], 'threat': y_pred[:,3], 'insult': y_pred[:,4], 'identity_hate': y_pred[:,5]})

output.head()

# output.to_csv('my_submission1.csv', index=False)

# print("Your submission was successfully saved!")
output.shape
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")