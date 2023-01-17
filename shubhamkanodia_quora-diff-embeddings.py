# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

from scipy.spatial.distance import cdist

from tensorflow.python.keras.models import Sequential

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, SpatialDropout1D

from keras.layers import *

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.callbacks import *

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import *

from keras.models import *

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import *

from keras.optimizers import *

import keras.backend as K

from keras.callbacks import *

import tensorflow as tf

import os

import time

import gc

import re

from unidecode import unidecode
df_train = pd.read_csv("../input/train.csv")

text_data = df_train["question_text"].values

target = df_train["target"].values

df_test = pd.read_csv("../input/test.csv")

test_data = df_test["question_text"].values

X_train, X_test, y_train, y_test = train_test_split(text_data,target, random_state = 23, test_size=0.0)

numWords = 20000

tokenizer = Tokenizer(num_words = numWords)

tokenizer.fit_on_texts(text_data)

x_train_tokens = tokenizer.texts_to_sequences(X_train)

x_test_tokens = tokenizer.texts_to_sequences(test_data)

# len_features = [len(x) for x in x_train_tokens]

# sum(x>50 for x in len_features)

pad = 'pre'

maxTokens = 50

x_train_pad = pad_sequences(x_train_tokens, maxlen=maxTokens, padding=pad,

                           truncating=pad)

x_test_pad = pad_sequences(x_test_tokens, maxlen=maxTokens, padding=pad,

                          truncating = pad)

y_pred_lst = []
# from tqdm import tqdm

# inverse_transform = dict(zip(tokenizer.word_index.values(),tokenizer.word_index.keys()))

# def convertTokensToString(tokens):

#     words = [inverse_transform[token] for token in tokens if token!=0]

#     return words

# x_train_tokens_words = [convertTokensToString(x) for x in tqdm(x_train_tokens)]

# x_test_tokens_words = [convertTokensToString(x) for x in tqdm(x_test_tokens)]
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def generate_embeddings(EMBEDDING_FILE):

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index

    nb_words = min(numWords,len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= numWords: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embed_size, embedding_matrix

def squash(x, axis=-1):

    # s_squared_norm is really small

    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)

    # return scale * x

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



# A Capsule Implement with Pure Keras

class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)



    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer='glorot_uniform',

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer='glorot_uniform',

                                     trainable=True)

    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]



        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)
def build_model():

    max_features = numWords

    maxlen=50

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

    x = SpatialDropout1D(rate=0.2)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True,

                               recurrent_initializer = orthogonal(gain=1.0,seed = 10000)))(x)

    x = CuDNNGRU(26, return_sequences=True)(x)

    x = Capsule(num_capsule = 10, dim_capsule = 10,

            routings=4,

            share_weights=True)(x)

    x = Flatten()(x)

    x = Dense(70, activation="relu")(x)

    x = Dropout(0.12)(x)

    x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    model.fit(x_train_pad,y_train,validation_split=0.05,epochs=2,batch_size=512)

    y_pred = model.predict(x_test_pad, verbose=1)

    y_pred_training = model.predict(x_train_pad, verbose=1)

    return (y_pred, y_pred_training)
import gc

EMBEDDING_FILES = ['../input/embeddings/glove.840B.300d/glove.840B.300d.txt',

                   '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',

                   '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt']

y_pred_lst=[]

y_pred_lst_training=[]

for EMBEDDING_FILE in EMBEDDING_FILES:

    embed_size, embedding_matrix = generate_embeddings(EMBEDDING_FILE)

    y_pred, y_pred_training = build_model()

    y_pred_lst.append(y_pred)

    y_pred_lst_training.append(y_pred_training)

    del embedding_matrix

    gc.collect()
# y_pred = y_pred_lst[len(y_pred_lst)-1]

# def get_thresh_f1(y_pred):

#     f1score_max = 0

#     thresh_max = 0

#     for thresh in np.arange(0.1, 0.501, 0.01):

#         thresh = np.round(thresh, 2)

#         if(f1score_max<f1_score(y_test, (y_pred>thresh).astype(int))):

#             f1score_max = f1_score(y_test, (y_pred>thresh).astype(int))

#             thresh_max = thresh

#     return f1score_max, thresh_max

# from sklearn.metrics import accuracy_score, f1_score

# get_thresh_f1(y_pred)

# ## (0.6697490870538343, 0.36) cudnn 26



##For final submission

##{glove : 0.29, wiki: 0.27}

#y_pred_final= (y_pred>0.36).astype(int).flatten()
train_arr = np.concatenate((y_pred_lst_training[0],y_pred_lst_training[1],y_pred_lst_training[2]), axis=1)

test_arr = np.concatenate((y_pred_lst[0],y_pred_lst[1],y_pred_lst[2]), axis=1)
from keras.utils import to_categorical

model = Sequential()

model.add(Dense(10, input_dim=3, activation='relu'))

model.add(Dense(4, input_dim=3, activation='relu'))

model.add(Dense(2, activation='softmax'))

# Compile model

y_train_categorical = to_categorical(y_train)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(train_arr,y_train_categorical,epochs=3,batch_size=256)

y_pred = model.predict(test_arr, verbose=1)

#10, 4
y_pred_final = np.argmax(y_pred,axis=1)
df_submission = pd.DataFrame({"qid":df_test["qid"].values, "prediction":y_pred_final})

df_submission = df_submission[["qid","prediction"]]

df_submission.to_csv("submission.csv",index=False)
# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_test, (y_pred>thresh).astype(int))
# y_pred = 0.33*y_pred_gnews + 0.33*y_pred_wiki + 0.33*y_pred_paragram
# max_features = numWords

# maxlen=50

# inp = Input(shape=(maxlen,))

# x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

# x = Bidirectional(CuDNNGRU(14, return_sequences=True))(x)

# x = CuDNNGRU(12, return_sequences=True)(x)

# x = GlobalMaxPool1D()(x)

# x = Dense(16, activation="relu")(x)

# x = Dropout(0.1)(x)

# x = Dense(1, activation="sigmoid")(x)

# model = Model(inputs=inp, outputs=x)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# print(model.summary())

# model.fit(x_train_pad,y_train,validation_split=0.05,epochs=2,batch_size=512)

# y_pred = model.predict(x_test_pad, verbose=1)