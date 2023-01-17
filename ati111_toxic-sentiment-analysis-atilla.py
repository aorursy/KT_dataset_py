!ls -al /kaggle/input/jigsaw-toxic-comment-classification-challenge
!pwd
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

#if len(K.tensorflow_backend._get_available_gpus()) > 0:

from keras.layers import CuDNNLSTM as LSTM

from keras.layers import CuDNNGRU as GRU





# Download the data:

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# Download the word vectors:

# http://nlp.stanford.edu/data/glove.6B.zip





# some configuration

MAX_SEQUENCE_LENGTH = 100

MAX_VOCAB_SIZE = 20000

EMBEDDING_DIM = 50

VALIDATION_SPLIT = 0.2

BATCH_SIZE = 128

EPOCHS = 5
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D
# load Glove 6B pre-trained 50--D word vectors

#print('Loading word vectors...')

word2vec = {}

#Open word vector text file

with open(os.path.join('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f: 

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

  for line in f:

    values = line.split()

    word = values[0]

    vec = np.asarray(values[1:], dtype='float32')

    word2vec[word] = vec

#print('Found %s word vectors in Glove.' % len(word2vec))
# prepare text samples and their labels

#print('Loading in comments...')



#train = pd.read_csv("../large_files/toxic-comment/train.csv")

train = pd.read_csv('/kaggle/input/dataset/train.csv')

test = pd.read_csv("/kaggle/input/dataset/test.csv")



#Get all sentences from train and test data

sentences_train = train["comment_text"].fillna("DUMMY_VALUE").values

sentences_test = test["comment_text"].fillna("DUMMY_VALUE").values



# Get train sentences labels

possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_targets = train[possible_labels].values

#test_targets = test[possible_labels].values
train.shape
# convert the sentences (strings) into integers

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE) #Create tokenizers (to vectorized text corpora)

tokenizer.fit_on_texts(sentences_train) #Updates internal vocabulary based on a list of texts (ie sentences_train)

sentences_train = tokenizer.texts_to_sequences(sentences_train) # Transforms each text in sequences_train to a sequence of integers.



#tokenizer.fit_on_texts(sentences_test)

sentences_test = tokenizer.texts_to_sequences(sentences_test)





# get word -> integer mapping number found by tokenizer in train sentences

word2idx = tokenizer.word_index

#print('Found %s unique tokens.' % len(word2idx))



# pad sequences so that we get a N x T matrix for all sequences as needed by keras

data_train = pad_sequences(sentences_train, maxlen=MAX_SEQUENCE_LENGTH)

data_test = pad_sequences(sentences_test, maxlen=MAX_SEQUENCE_LENGTH)



#print('Shape of train tensor:', data_train.shape)
data_train

# prepare embedding matrix  by keeping Glove 6B word found in sentences_train and sentences_test

# Result will be of (input_dim = MAX_VOCAB_SIZE, output_dim = EMBEDDING_DIM of glove 6B vectors)

#print('Filling pre-trained embeddings...')

num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx.items(): # word loop over unique words found by tokenizer

  if i < MAX_VOCAB_SIZE: #  check that vocab size keep < MAX_VOCAB_SIZE

    embedding_vector = word2vec.get(word) # return glove vector of word

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

      embedding_matrix[i] = embedding_vector
# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed (as )

embedding_layer = Embedding(

  num_words,          # input_dim = How large is the vocabulary ? 

  EMBEDDING_DIM,      # output_dim = How many numbers in the vector that you wish to return ?

  weights=[embedding_matrix], # weights: list of numpy arrays to set as initial weights of dim (input_dim, output_dim).

  input_length=MAX_SEQUENCE_LENGTH, # input_length = How many items are in the input feature vector that you need to transform ?

  trainable=False     

)
#print('Building model...')



# create an LSTM network with a single LSTM

input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))

x = embedding_layer(input_)

# x = LSTM(15, return_sequences=True)(x)

x = Bidirectional(LSTM(15, return_sequences=True))(x)

#x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)

output = Dense(len(possible_labels), activation="sigmoid")(x)



model = Model(input_, output)

model.compile(

  loss='binary_crossentropy',

  optimizer=Adam(lr=0.01),

  metrics=['accuracy']

)
#print('Training model...')

r = model.fit(

  data_train,

  train_targets,

  batch_size=BATCH_SIZE,

  epochs=EPOCHS,

  validation_split=VALIDATION_SPLIT

)
# plot loss history 

plt.plot(r.history['loss'], label='loss')

plt.plot(r.history['val_loss'], label='val_loss')

plt.legend()

plt.show()



# plot accuracies history

plt.plot(r.history['accuracy'], label='accuracy')

plt.plot(r.history['val_accuracy'], label='val_accuracy')

plt.legend()

plt.show()



model.predict(data_test, batch_size=1024, verbose=1)
import pandas as pd

sample_submission = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-dataset/sample_submission.csv")

test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-dataset/test.csv")

test_labels = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-dataset/test_labels.csv")

train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-dataset/train.csv")


aucs = []

for j in range(6):

    auc = roc_auc_score(targets[:,j], p[:,j])

    aucs.append(auc)

print(np.mean(aucs))
import pandas as pd

sample_submission = pd.read_csv("../input/sample_submission.csv")

test = pd.read_csv("../input/test.csv")

test_labels = pd.read_csv("../input/test_labels.csv")

train = pd.read_csv("../input/train.csv")