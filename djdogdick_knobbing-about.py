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
import matplotlib.pyplot as plt



from tqdm import tqdm # spits out a lil progress bar anytime you want to load a dataset



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.callbacks import Callback



# Useful for preprocessing

import string

import nltk

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import re



from sklearn.model_selection import train_test_split 
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', index_col='id')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id') # covariates you'll use for contest submmissions

train.head()
train.columns
plt.hist(train["target"])

plt.title("Distribution of Target Values")

plt.show()
# This could be useful if we want to regularize the target value with a beta prior, or use a weighted loss

plt.hist(train["toxicity_annotator_count"])

plt.title("Distribution of Toxicity Annotator Counts")

plt.show()
test_size = 10000

random_state = 2018
## split to train and val

train_df, val_df = train_test_split(train, 

        test_size=test_size, random_state=random_state) 



## fill up the missing values

train_X = train_df["comment_text"].fillna("_##_").values

val_X = val_df["comment_text"].fillna("_##_").values

test_X = test["comment_text"].fillna("_##_").values



## Preprocessing goes here



## Tokenize the sentences

## some config values 

embed_size = 300 # how big is each word vector

max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 70 # max number of words in a question to use



print("Tokenizing")

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

print("Padding")

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

train_y = train_df['target'].values > 0.5

val_y = val_df['target'].values > 0.5
def load_glove(word_index):

    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt' 

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.005838499,0.48782197

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 
embedding_matrix = load_glove(tokenizer.word_index)
# https://www.kaggle.com/yekenot/2dcnn-textclassifier

# It's kind of stupid that this works so well lol

def model_cnn(embedding_matrix):

    filter_sizes = [1,2,3,5]

    num_filters = 36



    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

    x = Reshape((maxlen, embed_size, 1))(x)



    maxpool_pool = []

    for i in range(len(filter_sizes)):

        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),

                                     kernel_initializer='he_normal', activation='elu')(x)

        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))



    z = Concatenate(axis=1)(maxpool_pool)   

    z = Flatten()(z)

    z = Dropout(0.1)(z)



    outp = Dense(1, activation="sigmoid")(z)



    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
model = model_cnn(embedding_matrix)
epochs = 5



histories = [None]*epochs

for epoch in range(epochs):

    histories[epoch] = model.fit(train_X, train_y, batch_size=512, epochs=1,

                  validation_data=(val_X, val_y), verbose=True)
pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
print(((pred_val_y >= 0.5) == val_y).mean())
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

# Code for neural attention layer. People have been using this same chunk of code for the past three competitions lol



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



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

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim

# Copied from my brodie https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate



class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())

def model_lstm_atten(embedding_matrix):

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = Attention(maxlen)(x)

    x = Dense(64, activation="relu")(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model2 = model_lstm_atten(embedding_matrix)
epochs = 5



histories = [None]*epochs

for epoch in range(epochs):

    histories[epoch] = model2.fit(train_X, train_y, batch_size=512, epochs=1,

                  validation_data=(val_X, val_y), verbose=True)
pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)



print(((pred_val_y > 0.5) == val_y).mean())