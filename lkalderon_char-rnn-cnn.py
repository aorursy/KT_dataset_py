import re

import sys

import numpy as np

import pandas as pd

import pickle

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

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras.callbacks import Callback, ReduceLROnPlateau

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

import tensorflow as tf



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Model

from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply

from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU, GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization

from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re,gc

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score,log_loss

import os

import pandas as pd
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train_data[classes].values



train_sentences = train_data["comment_text"].fillna("fillna").str.lower()

test_sentences = test_data["comment_text"].fillna("fillna").str.lower()
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



        self.aucs_tra.append(score_tra)

        self.aucs_val.append(score_val)

        print("\n ROC-AUC - epoch: %d - score_tra: %.6f - score_val: %.6f \n" % (epoch+1, score_tra, score_val))
max_features = 73

maxlen = 512



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,.!?:_@#$&"



def preproc_str(s):

    chs = s.encode('ascii', errors='ignore').decode('ascii', errors='ignore').lower()

    return chs #''.join([ch if ch in ALPHABET else ' ' for ch in chs])



train['comment_text'] = train_data.comment_text.fillna('').apply(preproc_str)

test['comment_text'] = test_data.comment_text.fillna('').apply(preproc_str)



list_sentences_train = train["comment_text"].fillna("").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_test = test["comment_text"].fillna("").values



tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

print(tokenizer.word_counts.items())

print('padding sequences')

X_train = {}

X_test = {}

X_train['text'] = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')

X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')



def flatten(l): return [item for sublist in l for item in sublist]

max_features = np.unique(flatten(X_train['text'])).shape[0] + 1

print('max_features_train:', max_features)

max_features_test = np.unique(flatten(X_test['text'])).shape[0] + 1

print('max_features_test:', max_features_test)



# print(train['comment_text'].values[239])

# print(test['comment_text'].values[239])
X_tra, X_val, y_tra, y_val = train_test_split(X_train['text'], y, train_size=0.95, random_state=233)

RocAuc = RocAucEvaluation(training_data=(X_tra, y_tra) ,validation_data=(X_val, y_val))
def _train_model(model, batch_size, train_x, train_y, val_x, val_y, callbacks_list):

    best_loss = -1

    best_weights = None

    best_epoch = 0



    current_epoch = 0



    while True:

        model.fit(train_x, train_y, batch_size=batch_size, epochs=1, callbacks=callbacks_list)

        y_pred = model.predict(val_x, batch_size=batch_size)



        total_loss = 0

        for j in range(6):

            loss = log_loss(val_y[:, j], y_pred[:, j])

            total_loss += loss



        total_loss /= 6.

        auc = roc_auc_score(val_y, y_pred)



        print("Epoch {0} loss {1} best_loss {2} roc_auc {3}".format(current_epoch, total_loss, best_loss, auc))



        if (np.isnan(total_loss)):

            break



        current_epoch += 1

        if total_loss < best_loss or best_loss == -1:

            best_loss = total_loss

            best_weights = model.get_weights()

            best_epoch = current_epoch

        else:

            if current_epoch - best_epoch == 5:

                break



    model.set_weights(best_weights)

    return model



def get_model_cnn(X_train):

    inp = Input(shape=(maxlen, ), name="text")

    x = Embedding(max_features, 16)(inp)

    x = SpatialDropout1D(0.2)(x)

    c1 = Conv1D(64, 3, activation="relu")(x)

    c1 = GlobalMaxPool1D()(c1)

    c2 = Conv1D(64, 5, activation="relu")(x)

    c2 = GlobalMaxPool1D()(c2)

    c3 = Conv1D(64, 9, activation="relu")(x)

    c3 = GlobalMaxPool1D()(c3)

    x = Dropout(0.2)(Concatenate()([c1,c2,c3]))

    x = Dropout(0.2)(Dense(128, activation="relu")(x))

    x = Dropout(0.2)(Dense(128, activation="relu")(x))

    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=[inp], outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'val_loss'])

    return model        



from keras.layers import Dense, Embedding, Input

from keras.layers import Bidirectional, Dropout, SpatialDropout1D, CuDNNGRU

from keras.models import Model

from keras.optimizers import RMSprop



def get_model_rnn(X_train):

    input_layer = Input(shape=(maxlen,), name="text")

    embedding_layer = Embedding(max_features, 16)(input_layer)

    #x = SpatialDropout1D(0.2)(embedding_layer)

    x = embedding_layer

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

    c1 = Dropout(0.3)(GlobalMaxPool1D()(x))

    output_layer = Dense(6, activation="sigmoid")(c1)



    model = Model(inputs=[input_layer], outputs=output_layer)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy', 'val_loss'])



    return model
print('start modeling')

batch_size = 128

epochs = 20



#checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

callbacks_list = [early, RocAuc]



model = get_model_rnn(X_tra)

# model = get_model_cnn(X_tra)

model = _train_model(model, batch_size, X_tra, y_tra, X_val, y_val, callbacks_list=callbacks_list)