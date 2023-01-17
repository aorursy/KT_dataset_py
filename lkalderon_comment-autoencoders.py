import re

import sys

import numpy as np

import pandas as pd

from pymagnitude import *

import matplotlib.pyplot as plt

%matplotlib inline

import gc

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import RNN, GRU, LSTM, Dense, Input, Embedding, Dropout, Activation, concatenate

from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import initializers, regularizers, constraints, optimizers, layers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU

from keras.callbacks import Callback

from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten

from keras.preprocessing import text, sequence

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.models import Model

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

import pandas as pd

import numpy as np

import pandas as pd

import os

import nltk

import re

from bs4 import BeautifulSoup

import urllib3

from sklearn.feature_extraction.text import TfidfVectorizer

import itertools

from sklearn import preprocessing

from scipy import sparse

from keras import backend as K # Importing Keras backend (by default it is Tensorflow)

from keras.layers import Input, Dense # Layers to be used for building our model

from keras.models import Model # The class used to create a model

from keras.optimizers import Adam

from keras.utils import np_utils # Utilities to manipulate numpy arrays

from tensorflow import set_random_seed # Used for reproducible experiments

from tensorflow import keras

import gc

import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential, Model

from keras.layers import InputLayer, Input, Embedding, Dense, Dropout, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, Conv1D, CuDNNLSTM, CuDNNGRU, TimeDistributed, Reshape, Permute, LocallyConnected1D, concatenate, ELU, Activation, add, Lambda, BatchNormalization, PReLU, MaxPooling1D, GlobalMaxPooling1D

from keras.optimizers import Adam

from keras import regularizers

#from kgutil.models.keras.base import DefaultTrainSequence, DefaultTestSequence

#from kgutil.models.keras.rnn import KerasRNN, load_emb_matrix

from copy import deepcopy

import inspect



import os
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train_data[classes].values



train_sentences = train_data["comment_text"].fillna("fillna").str.lower()

test_sentences = test_data["comment_text"].fillna("fillna").str.lower()



max_features = 100000

max_len = 150

embed_size = 300



tokenizer = Tokenizer(max_features)

tokenizer.fit_on_texts(list(train_sentences))



tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)

tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)



train_padding = pad_sequences(tokenized_train_sentences, max_len)

test_padding = pad_sequences(tokenized_test_sentences, max_len)



#max_len = 150

#https://github.com/plasticityai/magnitude

#!curl -s http://magnitude.plasticity.ai/glove+subword/glove.6B.300d.magnitude --output vectors.magnitude



#vecs_word2vec = Magnitude('http://magnitude.plasticity.ai/word2vec/heavy/GoogleNews-vectors-negative300.magnitude', stream=True, pad_to_length=max_len) 

vecs_glove = Magnitude('http://magnitude.plasticity.ai/glove+subword/glove.6B.300d.magnitude')

#vecs_fasttext = Magnitude('http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude', pad_to_length=max_len)

#vecs_elmo = Magnitude('http://magnitude.plasticity.ai/elmo/medium/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.magnitude', stream=True, pad_to_length=max_len)



#vectors = Magnitude(vecs_fasttext, vecs_glove) # concatenate word2vec with glove

word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, vecs_glove.dim))



from tqdm import tqdm_notebook as tqdm

for word, i in tqdm(word_index.items()):

    if i >= max_features:

        continue

    embedding_vector = vecs_glove.query(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector



gc.collect()
import pickle
from keras.layers import *



num_words = 100000

maxlen = 150

embed_dim = 300

latent_dim = 128

batch_size = 32

pad_seqs = train_padding



#### Encoder Model ####

encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')

emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Body-Word-Embedding', mask_zero=False)

# Word embeding for encoder (ex: Issue Body)

x = emb_layer(encoder_inputs)

state_h = GRU(latent_dim, name='Encoder-Last-GRU')(x)

encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')

seq2seq_encoder_out = encoder_model(encoder_inputs)

#### Decoder Model ####

decoded = RepeatVector(maxlen)(seq2seq_encoder_out)

decoder_gru = GRU(latent_dim, return_sequences=True, name='Decoder-GRU-before')

decoder_gru_output = decoder_gru(decoded)

decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')

decoder_outputs = decoder_dense(decoder_gru_output)

#### Seq2Seq Model ####

#seq2seq_decoder_out = decoder_model([decoder_inputs, seq2seq_encoder_out])

seq2seq_Model = Model(encoder_inputs,decoder_outputs )

seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')



history = seq2seq_Model.fit(pad_seqs, np.expand_dims(pad_seqs, -1),

          batch_size=batch_size,

          epochs=7,

          validation_split=0.12)



model_file = "autoencoder.sav"

with open(model_file,mode='wb') as model_f:

    pickle.dump(seq2seq_Model,model_f)
# #Feature extraction

# headlines = tokenizer.texts_to_sequences(data['headline'].values)

# headlines = pad_sequences(headlines,maxlen=maxlen)x = encoder_model.predict(headlines)

# #classifier

# X_train,y_train,X_test,y_test = x[msk],y[msk],x[~msk],y[~msk]

# lr = LogisticRegression().fit(X_train,y_train)

# lr.score(X_test,y_test)