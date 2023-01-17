import re

import sys

import gc

import numpy as np

import pandas as pd

from copy import deepcopy

import inspect

import os 



from pymagnitude import *

import matplotlib.pyplot as plt



import nltk



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

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, SimpleRNN

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.models import Model

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score



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

import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix

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





from keras.layers import Dense, LSTM, Bidirectional,Flatten

from keras.layers import Conv2D, MaxPool2D, Reshape,Embedding

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Concatenate

from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D



%matplotlib inline
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train_data[classes].values



train_sentences = train_data["comment_text"].fillna("fillna").str.lower()

test_sentences = test_data["comment_text"].fillna("fillna").str.lower()



max_features = 200000

max_len = 250

embed_size = 300



tokenizer = Tokenizer(max_features)

tokenizer.fit_on_texts(list(train_sentences))



tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)

tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)



train_padding = pad_sequences(tokenized_train_sentences, max_len)

test_padding = pad_sequences(tokenized_test_sentences, max_len)



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

    else:

        embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embed_size)



gc.collect()
# https://www.kaggle.com/yekenot/pooled-gru-fasttext



#Define a class for model evaluation

class RocAucEvaluation(Callback):

    def __init__(self, training_data=(),validation_data=()):

        super(Callback, self).__init__()

       

        self.X_tra, self.y_tra = training_data

        self.X_val, self.y_val = validation_data

        self.aucs_val = []

        self.aucs_tra = []

        

    def on_epoch_end(self, epoch, logs={}):                   

        y_pred_val = self.model.predict(self.X_val, verbose=0)

        score_val = roc_auc_score(self.y_val, y_pred_val)



        y_pred_tra = self.model.predict(self.X_tra, verbose=0)

        score_tra = roc_auc_score(self.y_tra, y_pred_tra)



        self.aucs_tra.append(score_val)

        self.aucs_val.append(score_tra)

        print("\n ROC-AUC - epoch: %d - score_tra: %.6f - score_val: %.6f \n" % (epoch+1, score_tra, score_val))
def BiLSTM_2DCNN(maxlen,max_features,embed_size,embedding_matrix,lstm_units=256):

    conv_filters = 32

    sequence_input = Input(shape=(maxlen,), dtype='int32')

    embedded_sequences = Embedding(max_features,embed_size,input_length=maxlen,weights=[embedding_matrix],trainable=False)(sequence_input)

    x = SpatialDropout1D(0.2)(embedded_sequences)

    x = Bidirectional(LSTM(lstm_units,return_sequences=True))(embedded_sequences)

    x = Dropout(0.1)(x)

    x = Reshape((2 * maxlen,lstm_units, 1))(x)

    x = Conv2D(conv_filters, (3, 3))(x)

    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.2)(x)

    x = Dense(64, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Flatten()(x)

    preds = Dense(6, activation='sigmoid')(x)

    model = Model(sequence_input, preds)

    return model
def bigru_cnn_1(

    data, target_shape,

    lr=1e-3,

    rnn_size=128, rnn_dropout=None, rnn_layers=1,

    conv_size=64, conv_activation=None,

    num_layers=[], num_activation='relu', num_dropout=None,

    mlp_layers=[], mlp_activation='relu', mlp_dropout=None,

    out_dropout=None,

    text_emb_dropout=0.2, text_emb_size=32, text_emb_file=None, text_emb_trainable=False, text_emb_rand_std=None

):

    if text_emb_file is not None:

        emb_weights = [load_emb_matrix(text_emb_file, data.text_tokenizer.word_index, data.text_voc_size, text_emb_size, rand_std=text_emb_rand_std)]

    else:

        emb_weights = None



    text_inp = Input(shape=[data.max_text_len], name='comment_text')



    inputs = [text_inp]



    seq = Embedding(data.text_voc_size, text_emb_size, weights=emb_weights, trainable=text_emb_trainable)(text_inp)

    seq = SpatialDropout1D(text_emb_dropout)(seq)



    for _ in range(rnn_layers):

        seq = Bidirectional(CuDNNGRU(rnn_size, return_sequences=True))(seq)

        if rnn_dropout is not None:

            seq = SpatialDropout1D(rnn_dropout)(seq)

    seq = Conv1D(conv_size, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(seq)

    seq = activation(conv_activation, seq)

    out = concatenate([GlobalMaxPool1D()(seq), GlobalAveragePooling1D()(seq)])



    if len(data.numeric_columns) > 0:

        num_inp = Input(shape=[len(data.numeric_columns)], name="numeric_columns__")

        inputs.append(num_inp)



        # Num MLP

        num = num_inp

        for layer_size in num_layers:

            if num_dropout is not None:

                num = Dropout(num_dropout)(num)

            num = Dense(layer_size, activation=None)(num)

            num = activation(num_activation, num)



        out = concatenate([out, num])



    # MLP

    for layer_size in mlp_layers:

        if mlp_dropout is not None:

            out = Dropout(mlp_dropout)(out)

        out = Dense(layer_size, activation=None)(out)

        out = activation(mlp_activation, out)



    # Output

    if out_dropout is not None:

        out = Dropout(out_dropout)(out)

    out = Dense(6, activation='sigmoid')(out)



    # Model

    model = Model(inputs, out)

    

    return model
X_tra, X_val, y_tra, y_val = train_test_split(train_padding, y, train_size=0.95, random_state=233)

RocAuc = RocAucEvaluation(training_data=(X_tra, y_tra) ,validation_data=(X_val, y_val))
model = BiLSTM_2DCNN(maxlen=max_len,

                     max_features=max_features,

                     embed_size=embed_size,

                     embedding_matrix=embedding_matrix,

                     lstm_units=256)





# go through epochs as long as accuracy on validation set increases

early_stopping = EarlyStopping(monitor='val_loss', 

                               patience=5,

                               mode='min')



# make sure that the model corresponding to the best epoch is saved

checkpointer = ModelCheckpoint(filepath='rnn_2dcnn.hdf5',

                               monitor='val_loss',

                               save_best_only=True,

                               verbose=0)



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.003))



model.fit(X_tra,

          y_tra,

          epochs=50,

          batch_size=64,

          shuffle=True,

          validation_data=(X_val, y_val),

          callbacks=[early_stopping, RocAuc, checkpointer]

      )