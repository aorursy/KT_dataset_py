from __future__ import absolute_import

import re

import sys

import numpy as np

import pandas as pd

from pymagnitude import *

import matplotlib.pyplot as plt

%matplotlib inline

import gc





import tensorflow as tf

from keras import backend as K

from keras import initializers

from keras import constraints

from keras import regularizers

from keras.engine import InputSpec, Layer

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
train_data = pd.read_csv('../input/cleaned_train.csv')

test_data = pd.read_csv('../input/cleaned_test.csv')



classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train_data[classes].values



train_sentences = train_data["comment_text"].fillna("fillna").str.lower()

test_sentences = test_data["comment_text"].fillna("fillna").str.lower()



max_features = 150000

max_len = 50

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
X_tra, X_val, y_tra, y_val = train_test_split(train_padding, y, train_size=0.95, random_state=233)

RocAuc = RocAucEvaluation(training_data=(X_tra, y_tra) ,validation_data=(X_val, y_val))
class Attention(Layer):

    def __init__(self,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True,  return_attention=True, **kwargs):

        """

        Keras Layer that implements an Attention mechanism for temporal data.

        Supports Masking.

        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

        # Input shape

            3D tensor with shape: `(samples, steps, features)`.

        # Output shape

            2D tensor with shape: `(samples, features)`.

        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

        The dimensions are inferred based on the output shape of the RNN.

        Note: The layer has been tested with Keras 2.0.6

        Example:

            model.add(LSTM(64, return_sequences=True))

            model.add(Attention())

            # next add a Dense layer (for classification/regression) or whatever...

        """

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())



        # in some cases especially in the early stages of training the sum may be almost zero

        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.

        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        att_weights = K.expand_dims(a)

       

        weighted_input = x * a

        restult = K.sum(weighted_input, axis=1)         

    

        if self.return_attention:

            return [result, att_weights]

        return result

    

#     def compute_output_shape(self, input_shape):

#         return input_shape[0], input_shape[-1]

        def compute_output_shape(self, input_shape):

            output_len = input_shape[2]

            if self.return_attention:

                return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]

            return (input_shape[0], output_len)

    

class AttentionWeightedAverage(Layer):

    """

    Computes a weighted average of the different channels across timesteps.

    Uses 1 parameter pr. channel to compute the attention value for a single timestep.

    """



    def __init__(self, return_attention=True, **kwargs):

        self.init = initializers.get('uniform')

        self.supports_masking = True

        self.return_attention = return_attention

        super(AttentionWeightedAverage, self).__init__(** kwargs)



    def build(self, input_shape):

        self.input_spec = [InputSpec(ndim=3)]

        assert len(input_shape) == 3



        self.W = self.add_weight(shape=(input_shape[2], 1),

                                 name='{}_W'.format(self.name),

                                 initializer=self.init)

        self.trainable_weights = [self.W]

        super(AttentionWeightedAverage, self).build(input_shape)



    def call(self, x, mask=None):

        # computes a probability distribution over the timesteps

        # uses 'max trick' for numerical stability

        # reshape is done to avoid issue with Tensorflow

        # and 1-dimensional weights

        logits = K.dot(x, self.W)

        x_shape = K.shape(x)

        logits = K.reshape(logits, (x_shape[0], x_shape[1]))

        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))



        # masked timesteps have zero weight

        if mask is not None:

            mask = K.cast(mask, K.floatx())

            ai = ai * mask

        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())

        weighted_input = x * K.expand_dims(att_weights)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:

            return [result, att_weights]

        return result



    def get_output_shape_for(self, input_shape):

        return self.compute_output_shape(input_shape)



    def compute_output_shape(self, input_shape):

        output_len = input_shape[2]

        if self.return_attention:

            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]

        return (input_shape[0], output_len)



    def compute_mask(self, input, input_mask=None):

        if isinstance(input_mask, list):

            return [None] * len(input_mask)

        else:

            return None
from __future__ import absolute_import



from keras.layers import *

from keras.layers.core import Activation

from keras.models import *

from keras.constraints import *

from keras.regularizers import *



def getModel0(input_shape, classes, num_words, emb_size, emb_matrix, emb_dropout=0.5,

              attention=0, dense=False, emb_trainable=False, gru=True):



    x_input = Input(shape=(input_shape,))

    

    emb = Embedding(num_words, emb_size, weights=[emb_matrix], trainable=emb_trainable, name='embs')(x_input)

    emb = SpatialDropout1D(emb_dropout)(emb)

    X = Dense(units=max_len, activation='relu')(emb)

    if gru:

        rnn1 = Bidirectional(CuDNNGRU(64, return_sequences=True))(emb) 

        rnn2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(rnn1)

    else:

        rnn1 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(emb)

        rnn2 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(rnn1)



    x = concatenate([rnn1, rnn2])



    if attention == 1: x, att_w = AttentionWeightedAverage()(x)

    elif attention == 2: x, att_w = Attention()(x)

    else: x = GlobalMaxPooling1D()(x)

    

    

    if dense: 

        x = Dense(32, activation='relu')(x)

        x = Dropout(0.3)(x)

    

    x_output = Dense(classes, activation='sigmoid')(x)

    return Model(inputs=x_input, outputs=x_output)
model = getModel0(input_shape=X_tra.shape[1],

                  classes=6,

                  num_words=173737,

                  emb_size=300,

                  emb_matrix=embedding_matrix,

                  emb_dropout=0.5,

                  attention=1,

                  dense=True,

                  emb_trainable=False,

                  gru=True)



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000015))



# saved_model = "weights_base.best.hdf5"

# checkpoint = ModelCheckpoint(saved_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

callbacks_list = [early, RocAuc]



model.fit(x=X_tra,

          y=y_tra,

          validation_data=(X_val, y_val),

          batch_size=128,

          epochs=2,

          shuffle=True,

          callbacks=callbacks_list, verbose=1)
layer_index = 6

intermediate_layer_model = Model(inputs=model.input,

                                 outputs=model.layers[layer_index].output[1])


def comment_attention_weights(comment, intermediate_layer_model=intermediate_layer_model, tokenizer=tokenizer):

    temp = tokenizer.texts_to_sequences([comment])

    comment_len = len(temp[0])

    temp_padded = pad_sequences(temp, max_len)    

    intermediate_output = intermediate_layer_model.predict(temp_padded)    

    print(intermediate_output[0][-comment_len:])

    print(np.sum(intermediate_output[0][:max_len-comment_len]))
comment_1 = "Unfortunately he was dead for ever"

comment_2 = "I want you dead for ever"

comment_attention_weights(comment=comment_1)

comment_attention_weights(comment=comment_2)
comment_1 = "He loves his black umbrella"

comment_2 = "he is a black man"

comment_attention_weights(comment=comment_1)

comment_attention_weights(comment=comment_2)
K.clear_session()

# del model

gc.collect()