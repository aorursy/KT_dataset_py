import sklearn
import tensorflow as tf
import numpy as np
import keras
import pandas as pd

import seaborn as sns
sns.set_style('whitegrid')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, LSTM, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, BatchNormalization
from keras.optimizers import *
from keras.initializers import *
from keras.activations import *
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

import datetime
import timeit

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

import re
import gc
## some config values 
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use
embed_size = 300
run_name = 'LSTM_GTU_Attention_class_balance'
train_df = pd.read_csv("../input/train.csv")

def clean_text(train_df):
    print("Cleaning text data...")

    mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", "colour": "color", "centre": "center", "favourite": "favorite", "travelling": "traveling", "counselling": "counseling", "theatre": "theater", "cancelled": "canceled", "labour": "labor", "organisation": "organization", "wwii": "world war 2", "citicise": "criticize", "youtu ": "youtube ", "Qoura": "Quora", "sallary": "salary", "Whta": "What", "narcisist": "narcissist", "howdo": "how do", "whatare": "what are", "howcan": "how can", "howmuch": "how much", "howmany": "how many", "whydo": "why do", "doI": "do I", "theBest": "the best", "howdoes": "how does", "mastrubation": "masturbation", "mastrubate": "masturbate", "mastrubating": 'masturbating', "pennis": "penis", "Etherium": "Ethereum", "narcissit": "narcissist", "bigdata": "big data", "2k17" : "2017", "2k18": "2018", "qouta": "quota", "exboyfriend" : "ex boyfriend", "airhostess" : "air hostess", "whst": "what", "watsapp": "whatsapp", "demonitisation": "demonetization", "demonitization": "demonetization", "demonetisation": "demonetization"}

    def _get_mispell(mispell_dict):
        mispell_re = re.compile("(%s)" % "|".join(mispell_dict.keys()))
        return mispell_dict, mispell_re

    mispellings, mispellings_re = _get_mispell(mispell_dict)
    def replace_typical_misspell(text):
        def replace(match):
            return mispellings[match.group(0)]
        return mispellings_re.sub(replace, text)

    # Lower the text
    train_df["question_text"] = train_df["question_text"].str.lower()

    # Clean numbers
    train_df["question_text"] = train_df["question_text"].str.replace(r"[0-9]{5,}", r"#####")
    train_df["question_text"] = train_df["question_text"].str.replace(r"[0-9]{4}", r"####")
    train_df["question_text"] = train_df["question_text"].str.replace(r"[0-9]{3}", r"###")
    train_df["question_text"] = train_df["question_text"].str.replace(r"[0-9]{2}", r"##")
    train_df["question_text"] = train_df["question_text"].str.replace(r"[0-9]*\.[0-9]*", r"##")

    # Clean spellings
    train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    
    # Clean the text
    train_df["question_text"] = train_df["question_text"].str.replace(r"([^\w\s\'\"])", r" \1 ")
    train_df["question_text"] = train_df["question_text"].str.replace(r"\s{2,}", r" ")
    
    return train_df

train_df = clean_text(train_df)
def data(train_df):
    
    #train_df = pd.read_csv("input/train_clean.csv")

    X = train_df["question_text"].values
    y = train_df["target"].values
     
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))
    X = tokenizer.texts_to_sequences(X)

    ## Pad the sentences 
    X = pad_sequences(X, maxlen=maxlen)
    
    # Make a small train test split to somehow evaluate the model accuracy. 
    # As this data is noisy and this won't be the same as the test set in the 
    # competition, I'm not really sure if this is a great idea, but we'll run with it
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42, \
                                                        shuffle = True, stratify = y)
    
    ###
    # Embedding loading
    ###
    
    #####
    ### GLOVE
    #####
    
    word_index = tokenizer.word_index
    
    print("Loading GloVe embedding...")
    
    EMBEDDING_FILE = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype="float32")
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_glove = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix_glove[i] = embedding_vector
    
    del embeddings_index, all_embs, emb_mean, emb_std, nb_words, embedding_vector
    
    print("GloVe embedding loaded...")
    
    ######
    ### PARAGRAM Embedding
    ######
    
    print("Loading Paragram embedding...")
    
    EMBEDDING_FILE = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype="float32")
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors="ignore") if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]
    #print(emb_mean,emb_std,"para")

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_para = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix_para[i] = embedding_vector
    
    del embeddings_index, all_embs, emb_mean, emb_std, nb_words, embedding_vector 
    print("Paragram embedding loaded...")
    
    print("Concatenating embedding matrices...")
    
    embedding_matrix = np.mean([embedding_matrix_para, embedding_matrix_glove], axis = 0)
    
    del embedding_matrix_para, embedding_matrix_glove
    
    print("Data loading done...")
    
    return X_train, X_test, y_train, y_test, embedding_matrix #, max_features, maxlen, embed_size    


####
# Extra classes for the training
####

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")

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

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode="triangular",
                 gamma=1., scale_fn=None, scale_mode="cycle"):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = "iterations"
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
        if self.scale_mode == "cycle":
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

        self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault("iterations", []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
# Data loading...

X_train, X_test, y_train, y_test, embedding_matrix = data(train_df)
# Model based on hyperopt since I'm going crazy
import hyperopt
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
# for better class weights see : https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
from sklearn.utils import class_weight
import json

# hp.choice('dense_layers', np.arange(20, 80, 5, dtype=int))

space = {'k_folds' : 3,
         'dropout_1d_rate' : hp.uniform('dropout_1d_rate', 0, 1),
         'use_LSTM_layer' : hp.choice('use_LSTM_layer', [{'use_layer' : 'no'}, \
                                                         {'use_layer' : 'yes', \
                                                          'LSTM_layers' : hp.quniform('LSTM_layers', 20, 80, 13),
                                                          'dropout_rate_lstm' : hp.uniform('dropout_rate_lstm', 0, 1)}]),
         'use_GRU_layer' : hp.choice('use_GRU_layer', [{'use_layer' : 'no'}, \
                                                       {'use_layer' : 'yes', \
                                                        'GRU_layers' : hp.quniform('GRU_layers', 20, 80, 13),
                                                        'dropout_rate_gru' : hp.uniform('dropout_rate_gru', 0, 1)}]),
         'dense_layers' : hp.quniform('dense_layers', 20, 80, 13),
         'dropout_rate_dense' : hp.uniform('dropout_rate_dense', 0, 1),
         'batch_size' : hp.choice('batch_size', [512, 1024, 2048]),
         'epochs' : hp.choice('epochs', [2,3,4])
         #,'random_seed' : # leave random seed tuning for last...
        }

# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)

def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2

###
# REASONING : So as I've seen in the CV vs LB score on this competition, it seems that having more 
# epochs and more folds definitely increases the CV accuracy and loss, but gives a very variable
# score on the LB. So the idea is to see which configuration quite quickly gives a good loss/accuracy, 
# and then try that out on the LB

# What is a good metric to measure the loss by? Accuracy of the here defined test function is probably
# not too bad, because it's truly a holdout (the test set that has been split out), 
# but maybe it should be a bit bigger than 10% (made it 15%)

# Version 2 (which includes switching on and off of layers, and is posted to kaggle)
# changes the above concern by not optimizing on the accuracy but on the F1 from the hold 
# out test set, which is far closer to what happens on Kaggle.

def objective(params):

    max_features = 95000 
    maxlen = 70 
    embed_size = 300
    
    print('Currently searching over : {}'.format(params))
    
    ###
    # kfolds
    ###
    
    kfold = StratifiedKFold(n_splits=params['k_folds'], random_state=10, shuffle=True)
    
    ###
    # define model, model based on a comment in the discussion section, which I couldn't
    # find anymore, so thanks someone!
    ###
    
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate = params['dropout_1d_rate'])(x)
    if params['use_LSTM_layer']['use_layer'] == 'yes':
        x = Bidirectional(CuDNNLSTM(units = int(round(params['use_LSTM_layer']['LSTM_layers'])), return_sequences=True, 
                                    kernel_initializer=glorot_normal(seed=12300), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)
        #x = Dropout(rate = params['use_LSTM_layer']['dropout_rate_lstm'])(x)
    if params['use_GRU_layer']['use_layer'] == 'yes':
        x = Bidirectional(CuDNNGRU(units = int(round(params['use_GRU_layer']['GRU_layers'])), return_sequences=True, 
                                   kernel_initializer=glorot_normal(seed=12300), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)
        #x = Dropout(rate = params['use_GRU_layer']['dropout_rate_gru'])(x)

    x = Attention(maxlen)(x)
    x = Dense(units = int(round(params['dense_layers'])), activation="linear", kernel_initializer=glorot_normal(seed=12300))(x)
    x = Dropout(rate = params['dropout_rate_dense'])(x)
    x = BatchNormalization()(x)
    #x = Activation("relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    # Use all GPUs
    model.compile(loss="binary_crossentropy", optimizer=Adam(), \
                  metrics = ["accuracy"])

    filepath="weights_best.h5"
    # Checkpoint not really necessary since it only improves the run time at this moment
    #checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=2, save_best_only=True, mode="min")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=2, verbose=2, mode="auto")
    callbacks = [earlystopping, reduce_lr]
    
    ###
    # Apply the folds over the model
    ###

    for i, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):
        Xk_train, Xk_val, Yk_train, Yk_val = X_train[train_index], X_train[valid_index], y_train[train_index], y_train[valid_index]

        print("Currently in fold {}/{}".format(i+1, params['k_folds']))
        model.fit(Xk_train, Yk_train, batch_size=params['batch_size'], epochs=params['epochs'], \
                           validation_data=(Xk_val, Yk_val), callbacks=callbacks, class_weight=class_weights)
        #model.load_weights(filepath) 

    score, acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Also add f1 and find optimal threshold,
    # this way we can compute f1 and optimize for that...
    
    pred_val_y = model.predict([X_test], batch_size=params['batch_size'], verbose=0)
    f1, threshold = f1_smart(np.squeeze(y_test), np.squeeze(pred_val_y))
    
    print('Accuracy : {:5f}, Optimal F1 : {:5f}, at threshold : {:5f}'.format(acc, f1, threshold))
    
    ### Save to file what we've just done...
    # I know this is redundant since we're already saving trials, but I had a notebook
    # crash om me sometimes and this saves the intermediate results, so definitely helps
    
    ## https://stackoverflow.com/questions/33054527/python-3-5-typeerror-a-bytes-like-object-is-required-not-str-when-writing-t
    with open("run_{}.txt".format(run_name),"a") as f:
        print(params, file=f)
        print('\n Accuracy : {}, F1 : {}, optimal threshold : {}'.format(acc, f1, threshold), file = f)  
        f.close() 
    
    return {"loss": -f1, "status": STATUS_OK, "model": model} # "loss" : -acc

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=1) #200

print(hyperopt.space_eval(space, best))
print(trials.best_trial)
import hyperopt
print(hyperopt.space_eval(space, best))
print(trials.best_trial)
# remove the keras model since this one does not play nicely with the pickle
# cleaned_trials_trials = []
# for trial in trials.trials:
#     del trial['result']['model']
#     cleaned_trials_trials.append(trial)
# from https://github.com/hyperopt/hyperopt/issues/267
# & https://github.com/hyperopt/hyperopt/wiki/FMin
# for some nice postprocesing and plotting of the results see:
# https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

# Just saving the results here which should be interprable from the results
# can't pickle the whole trials file because it contains Keras models and 
# those can't be pickled apparently :((((

# import pickle

# trials_trials = trials.trials
# trials_losses = trials.losses()
# trials_statuses = trials.statuses()

# pickle.dump([cleaned_trials_trials, trials_losses, trials_statuses, space], open("hyperopt_result_{:%Y-%m-%d %H:%M:%S}.p".format(datetime.datetime.now()), "wb"))
# # trials = pickle.load(open("myfile.p", "rb")) # for later reloading if necessary
# file = open("results_{:%Y-%m-%d %H:%M:%S}.txt".format(datetime.datetime.now()),"w") 
# print(hyperopt.space_eval(space, best), file = file) 
# print(trials.best_trial, file = file)

# file.close() 
