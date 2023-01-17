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

from sklearn.metrics import roc_auc_score
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
val_size = 10000

random_state = 2018
## split to train and val

train_df, val_df = train_test_split(train, 

        test_size=val_size, random_state=random_state) 



## fill up the missing values

train_X = train_df["comment_text"].values

val_X = val_df["comment_text"].values

test_X = test["comment_text"].values



## Preprocessing ought to go here



## Tokenize the sentences

## some config values 

embed_size = 300 # how big is each word vector

max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 70 # max number of words in a question to use



print("Tokenizing") # This part takes a long time. Sorry!

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
embedding_matrix = load_glove(tokenizer.word_index) # also takes a long time. can't really get around this
# https://www.kaggle.com/yekenot/2dcnn-textclassifier

# It's kind of stupid that this works so well lol. That's CNNs for you

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



fast_model = model_cnn(embedding_matrix)

fast_model.fit(train_X, train_df["target"] > 0.5, batch_size=512, epochs=1,

          validation_data=(val_X, val_y), verbose=True)

pred_val_y = fast_model.predict([val_X], batch_size=1024, verbose=0).T[0]

roc_auc_score(y_true=(val_df["target"] > 0.5), y_score=pred_val_y)
def get_roc_auc(model, X, target): #slightly more convenient than the two-liner used previously

    pred_val_y = model.predict([X], batch_size=1024, verbose=0).T[0]

    return roc_auc_score(y_true=(target > 0.5), y_score=pred_val_y)



get_roc_auc(fast_model, val_X, val_df["target"])
def get_false_pos_rate(y_true, y_pred, thresh=0.5):

    return (y_pred > thresh)[1 - y_true].mean()



def get_true_pos_rate(y_true, y_pred, thresh=0.5):

    return 1 - y_true[y_pred > thresh].mean()



thresholds = np.arange(0.0, 1.0, 0.01)

fp_rates = [get_false_pos_rate(val_df["target"] > 0.5, pred_val_y, thresh) for thresh in thresholds]

tp_rates = [ get_true_pos_rate(val_df["target"] > 0.5, pred_val_y, thresh) for thresh in thresholds]



print(list(zip(thresholds, fp_rates, tp_rates))[0:5])
plt.plot(fp_rates, tp_rates)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Plot")

plt.show()
import keras.backend as K



def custom_loss(p_reviewers, y_pred, n_reviewers):

    """ Treats every instance of a review as a separate datapoint, rather than giving each comment equal weight.

    Use this if you assume that reviewers are i.i.d. and that the number of reviewers is independent of the 

    comment's contents.

    

    Requires: 

    - p_reviewers: n-length array. p_reviewers[i] is float representing 

    proportion of reviewers who labelled comment i as "toxic"

    - y_pred: n-length array of floats, where y_pred[i] is model's predicted probability that 

    comment i is toxic

    - n_reviewers: n-length array. n_reviewers[i] is int representing 

    total number of reviewers who labelled given comment as "toxic" or "not toxic"

    Returns: float"""

    # Maybe K.mean results in some efficiency? 

    return -1*K.mean(K.log(y_pred) * n_reviewers * p_reviewers + 

                 K.log(1-y_pred) * n_reviewers * (1-p_reviewers))

from functools import partial



# Same CNN model as before, but with the custom loss fn. 

# Quite slow to train and actually gets lower ROC-AUC

def model_cnn_custom_loss(embedding_matrix):

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

    n_reviewers_inp = Input(shape=(1,))



    model = Model(inputs=[inp, n_reviewers_inp], outputs=outp)

    # partial loss function composition - this is necessary cuz keras loss functions require exactly 

    # two arguments: y_true and y_pred. This helps us get around that

    custom_loss_partial = partial(custom_loss, n_reviewers=train_df["toxicity_annotator_count"].values)

    model.compile(loss=custom_loss_partial, optimizer='adam', metrics=['accuracy'])

    return model



# This takes 15 MINUTES to train for 1 epoch. Pretty ridiculous

# slow_model = model_cnn_custom_loss(embedding_matrix)

# slow_model.fit(x=[train_X, train_df["toxicity_annotator_count"].values], y=train_y, 

#                batch_size=512, epochs=1,

#                validation_data=([val_X, val_df["toxicity_annotator_count"]], val_y), verbose=True)



# pred_val_y = slow_model.predict([val_X, np.array([1]*len(val_X))], batch_size=1024, verbose=0).T[0]

# print(roc_auc_score(y_true=(val_df["target"] > 0.5), y_score=pred_val_y))
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

# Code for neural attention layer. People have been using this same chunk of code for the past three competitions lol

# I'd recommend ignoring this completely tbh.



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
epochs = 1



histories = [None]*epochs

for epoch in range(epochs):

    # model saves weights from previous call to .fit. Can access weights in history object returned by .fit

    histories[epoch] = model2.fit(train_X, train_y, batch_size=512, epochs=1,

                  validation_data=(val_X, val_y), verbose=True)
# pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

# roc_auc_score(y_true=(val_df["target"] > 0.5), y_score=pred_val_y)

get_roc_auc(model2, val_X, val_df["target"])
predictions = model2.predict([test_X], batch_size=1024, verbose=0)



submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': predictions

})



submission.to_csv('submission.csv', index=False)