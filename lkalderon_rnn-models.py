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

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, SimpleRNN

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



max_features = 150000

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

vecs_fasttext = Magnitude('http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude', pad_to_length=max_len)

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



        self.aucs_tra.append(score_tra)

        self.aucs_val.append(score_val)

        print("\n ROC-AUC - epoch: %d - score_tra: %.6f - score_val: %.6f \n" % (epoch+1, score_tra, score_val))



def recall(y_true, y_pred):    

    """

    Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of

    how many relevant items are selected.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def precision(y_true, y_pred):    

    """

    Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of

    how many selected items are relevant.

    Source

    ------

    https://github.com/fchollet/keras/issues/5400#issuecomment-314747992

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision





def f1(y_true, y_pred):

    

    """Calculate the F1 score."""

    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    return 2 * ((p * r) / (p + r))





def accuracy(y_true, y_pred):

    return K.mean(K.equal(y_true, K.round(y_pred)), axis=1)
class Plots:

    def plot_history(history):

        loss = history.history['loss']

        val_loss = history.history['val_loss']

        x = range(1, len(val_loss) + 1)



        plt.plot(x, loss, 'b', label='Training loss')

        plt.plot(x, val_loss, 'r', label='Validation loss')

        plt.title('Training and validation loss')

        plt.legend()



    def plot_roc_auc(train_roc, val_roc):

        x = range(1, len(val_roc) + 1)



        plt.plot(x, train_roc, 'b', label='Training RocAuc')

        plt.plot(x, val_roc, 'r', label='Validation RocAuc')

        plt.title('Training and validation RocAuc')

        plt.legend()
X_tra, X_val, y_tra, y_val = train_test_split(train_padding, y, train_size=0.90, random_state=233)

RocAuc = RocAucEvaluation(training_data=(X_tra, y_tra) ,validation_data=(X_val, y_val))
input_layer = Input(shape=(max_len, ))

X = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_layer)

X = SimpleRNN(units=128, activation="relu")(X)

X = Dense(6, activation="sigmoid")(X)

model = Model(inputs=input_layer, outputs=X)



model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', 'binary_crossentropy'])



saved_model = "weights_base.best.hdf5"



checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

callbacks_list = [checkpoint, early, RocAuc]



batch_sz = 64

epoch = 20



model.fit(X_tra,

          y_tra,

          validation_data=(X_val, y_val),

          batch_size=batch_sz,

          epochs=epoch,

          callbacks=callbacks_list,

          shuffle=True,

          verbose=1)
input_layer = Input(shape=(max_len, ))

X = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_layer)

X = Bidirectional(CuDNNGRU(128))(X)

X = Dense(6, activation="sigmoid")(X)

model = Model(inputs=input_layer, outputs=X)



model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', 'binary_crossentropy'])



saved_model = "weights_base.best.hdf5"



checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

callbacks_list = [checkpoint, early, RocAuc]



batch_sz = 128

epoch = 20



model.fit(X_tra,

          y_tra,

          validation_data=(X_val, y_val),

          batch_size=batch_sz,

          epochs=epoch,

          callbacks=callbacks_list,

          shuffle=True,

          verbose=1)
input_layer = Input(shape=(max_len, ))

X = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_layer)

X = Dense(units=max_len, activation='relu')(X)

X = BatchNormalization()(X)

X = Bidirectional(CuDNNGRU(128))(X)

X = Dense(6, activation="sigmoid")(X)

model = Model(inputs=input_layer, outputs=X)



model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', 'binary_crossentropy'])



saved_model = "weights_base.best.hdf5"



checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks_list = [checkpoint, early, RocAuc]



batch_sz = 128

epoch = 20



model.fit(X_tra,

          y_tra,

          validation_data=(X_val, y_val),

          batch_size=batch_sz,

          epochs=epoch,

          callbacks=callbacks_list,

          shuffle=True,

          verbose=1)
input_layer = Input(shape=(max_len, ))

X = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_layer)

# Embedding projection Layer before the RNN

X = Dense(units=max_len, activation='relu')(X)

# X = Dropout(0.2)(X)

X = BatchNormalization()(X)

X = Bidirectional(CuDNNGRU(128))(X)

# MLP on top of BiGRU

X = Dense(256, activation='relu' )(X)

# X = Dense(100, activation='relu' )(X)

X = Dense(6, activation="sigmoid")(X)

model = Model(inputs=input_layer, outputs=X)



model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', 'binary_crossentropy'])



saved_model = "weights_base.best.hdf5"



checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks_list = [checkpoint, early, RocAuc]



batch_sz = 128

epoch = 20



model.fit(X_tra,

          y_tra,

          validation_data=(X_val, y_val),

          batch_size=batch_sz,

          epochs=epoch,

          callbacks=callbacks_list,

          shuffle=True,

          verbose=1)
input_layer = Input(shape=(max_len, ))

X = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_layer)

# Embedding projection Layer before the RNN

X = Dense(units=max_len, activation='relu')(X)

# X = Dropout(0.2)(X)

X = BatchNormalization()(X)

x_state, x_fwd, x_bwd = Bidirectional(CuDNNGRU(128, return_sequences=True, return_state=True))(X)

X = concatenate([x_fwd, x_bwd])

# MLP on top of BiGRU

# X = Reshape((2 * max_len,128, 1))(X)

X = Dense(units=256, activation='relu')(X)

X = Dense(6, activation="sigmoid")(X)

model = Model(inputs=input_layer, outputs=X)



model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', 'binary_crossentropy'])



saved_model = "weights_base.best.hdf5"



checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks_list = [checkpoint, early, RocAuc]



batch_sz = 128

epoch = 20



model.fit(X_tra,

          y_tra,

          validation_data=(X_val, y_val),

          batch_size=batch_sz,

          epochs=epoch,

          callbacks=callbacks_list,

          shuffle=True,

          verbose=1)

model.summary()
target_shape = 6

lr=0.0003

rnn_dropout=None

rnn_layers=[128, 64]

mlp_layers=[70]

mlp_dropout=0.1

text_emb_dropout=0.0

text_emb_size=300



model = Sequential()

model.add(InputLayer(name='comment_text', input_shape=[max_len]))

model.add(Embedding(max_features, text_emb_size, weights=[embedding_matrix], trainable=False))

model.add(Dropout(text_emb_dropout))



for layer_size in rnn_layers:

    #Fast LSTM implementation backed by CuDNN. Can only be run on GPU, with the TensorFlow backend.

    model.add(Bidirectional(CuDNNLSTM(layer_size, return_sequences=True)))

    if rnn_dropout is not None:

        model.add(SpatialDropout1D(rnn_dropout))



model.add(GlobalMaxPool1D())

for layer_size in mlp_layers:

    model.add(Dense(layer_size, activation="relu"))

    model.add(Dropout(mlp_dropout))

model.add(Dense(6, activation="sigmoid"))



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000015))



saved_model = "weights_base.best.hdf5"

checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=3)

callbacks_list = [checkpoint, early, RocAuc]



model.fit(x=X_tra,

          y=y_tra,

          validation_data=(X_val, y_val),

          batch_size=128,

          epochs=80,         

          callbacks=callbacks_list, verbose=1)
K.clear_session()

del model

gc.collect()
# model.summary() # Print a description of the model.

# Plots.plot_roc_auc(RocAuc.aucs_tra, RocAuc.aucs_val)

Plots.plot_history(model.history)
test_values = model.predict([test_padding], batch_size=1024, verbose=1)

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission[classes] = test_values

sample_submission.to_csv('/submission.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np



sample_submission.to_csv('submission.csv', index=False)



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='submission.csv')
from keras.models import Sequential

from keras.layers import Dense

from keras.utils.vis_utils import plot_model



plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

from IPython.display import Image

Image(retina=True, filename='model.png')
!pip install ann_visualizer

!pip install graphviz

!pip install h5py

from ann_visualizer.visualize import ann_viz

# create model

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ann_viz(model, title="Artificial Neural network - Model Visualization")
from keras.models import load_model

model = load_model('model.h5')