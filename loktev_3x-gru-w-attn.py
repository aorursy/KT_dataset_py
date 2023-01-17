num_words=50000

max_len=64



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import operator

import tensorflow as tf

import keras.backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, concatenate, Dense, Flatten, Embedding, GRU, CuDNNGRU, LSTM, CuDNNLSTM, SpatialDropout1D, Dropout, Bidirectional, Conv1D, Activation, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, RepeatVector, Permute

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras.optimizers import Adam

from tensorflow.keras import utils

from keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

tqdm.pandas()

import matplotlib.pyplot as plt

import tracemalloc

%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab

def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    

    return x

def clean_numbers(x):



    x = re.sub('[0-9]{5,}', ' huge number ', x)

    x = re.sub('[0-9]{4}', ' year ', x)

    x = re.sub('[0-9]{3}', ' number ', x)

    x = re.sub('[0-9]{2}', ' number ', x)

    return x



def clean_more(x):

    x=re.sub('\s+', ' ', x).strip()

    regex = re.compile('[^a-zA-Z] ')

    #First parameter is the replacement, second parameter is your input string

    return regex.sub('', x)

    









def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





mispell_dict = {'colour':'color',

                'centre':'center',

                'didnt':'did not',

                'doesnt':'does not',

                'isnt':'is not',

                'shouldnt':'should not',

                'favourite':'favorite',

                'travelling':'traveling',

                'counselling':'counseling',

                'theatre':'theater',

                'cancelled':'canceled',

                'labour':'labor',

                'organisation':'organization',

                'citicise':'criticize',

                }

mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)



train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_more(x))

train["question_text"] = train["question_text"].progress_apply(lambda x: x.lower())

train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

sentences = train["question_text"].apply(lambda x: x.split())

#to_remove = ['a','to','of','and']

#sentences = [[word.lower() for word in sentence if not word.lower() in to_remove] for sentence in tqdm(sentences)]





test["question_text"] = test["question_text"].progress_apply(lambda x: clean_text(x))

test["question_text"] = test["question_text"].progress_apply(lambda x: clean_numbers(x))

test["question_text"] = test["question_text"].progress_apply(lambda x: clean_more(x))

test["question_text"] = test["question_text"].progress_apply(lambda x: x.lower())

test["question_text"] = test["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

tsentences = test["question_text"].apply(lambda x: x.split())

#to_remove = ['a','to','of','and']

#tsentences = [[word.lower() for word in sentence if not word.lower() in to_remove] for sentence in tqdm(tsentences)]



vocab = build_vocab(list(sentences)+list(tsentences))

def wordindex(vocab,n):

  word_index={}

  sorted_v = sorted(vocab.items(), key=operator.itemgetter(1))[::-1]

  for i in range(n-3):

    word_index[sorted_v[i][0]]=i+3

  return(word_index)



word_index=wordindex(vocab,num_words)



def zif(word):

  ans=2

  if (word in word_index):

    ans=word_index[word]

  return ans



x_train = [[zif(word) for word in sentence] for sentence in tqdm(sentences)]

x_train=pad_sequences(x_train, maxlen=max_len)







x_test = [[zif(word) for word in sentence] for sentence in tqdm(tsentences)]

x_test=pad_sequences(x_test, maxlen=max_len)

print({k: vocab[k] for k in list(vocab)[:5]})
def wordindex(vocab,n):

  word_index={}

  sorted_v = sorted(vocab.items(), key=operator.itemgetter(1))[::-1]

  for i in range(n-3):

    word_index[sorted_v[i][0]]=i+3

  return(word_index)



word_index=wordindex(vocab,num_words)



def zif(word):

  ans=2

  if (word in word_index):

    ans=word_index[word]

  return ans



x_train = [[zif(word) for word in sentence] for sentence in tqdm(sentences)]

x_train=pad_sequences(x_train, maxlen=max_len)







x_test = [[zif(word) for word in sentence] for sentence in tqdm(tsentences)]

x_test=pad_sequences(x_test, maxlen=max_len)



y_train=train['target']
del train

del sentences

del tsentences
embedding_matrix_Glove = np.zeros((num_words, 300))

with open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', 'r') as f:

    for line in tqdm(f):

        values = line.split()

        word = values[0]

        if (word in word_index):

            try:

                word_vector = np.asarray(values[1:], dtype='float32')        

            except ValueError:

                pass  # do nothing!

            else:

                embedding_matrix_Glove[word_index[word]] = word_vector
embedding_matrix_Wiki = np.zeros((num_words, 300))

with open('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec', 'r') as f:

    for line in tqdm(f):

        values = line.split()

        word = values[0]

        if (word in word_index):

            try:

                word_vector = np.asarray(values[1:], dtype='float32')        

            except ValueError:

                pass  # do nothing!

            else:

                embedding_matrix_Wiki[word_index[word]] = word_vector
len(y_train)
def f1(y_true, y_pred):

    y_pred = K.round(y_pred+0.15)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)
class Netn:  

    def __init__(self,em):

        tweet_input = Input(shape=(max_len,), dtype='int32')

        tweet_encoder = Embedding(num_words, 300, input_length=max_len,

                          weights=[em], trainable=False)(tweet_input)

        X = SpatialDropout1D(0.1)(tweet_encoder)

        X = Bidirectional(CuDNNGRU(64, return_sequences=True))(X)

        activations = CuDNNGRU(64, return_sequences=True)(X)

        # compute importance for each step

        attention = Dense(1, activation='tanh')(activations)

        attention = Flatten()(attention)

        attention = Activation('softmax')(attention)

        attention = RepeatVector(64)(attention)

        attention = Permute([2, 1])(attention)

        attention = Dropout(0.5)(attention)

        X = concatenate([activations, attention])

        x=CuDNNGRU(64, return_sequences=True)(X)

        a=GlobalMaxPooling1D()(x)

        b=GlobalAveragePooling1D()(x) 

        x=concatenate([a,b])

        x=Dense(64, activation='relu')(x)

        x=Dropout(0.4)(x)

        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=[tweet_input], outputs=[output])

        self.model.summary()

    

    def unfreeze(self):

        self.model.layers[1].trainable = True

  

    def fit(self,b1,b2,epp,bs,**data):

        self.model.compile(**data)

        

        x_t=x_train[b1:b2]

        y_t=y_train[b1:b2]

        if (epp>0):

            filepath='tmp.hd5'

            cp=ModelCheckpoint(filepath, monitor="val_f1",verbose=1, save_best_only=True,mode='max')

            history=self.model.fit(x_t, 

                    y_t, 

                    epochs=epp,

                    batch_size=bs,

                    callbacks=[cp],

                    validation_split=0.1)

            self.model.load_weights(filepath)

            plt.plot(history.history['f1'], label='f1 train')

            plt.plot(history.history['val_f1'], label='f1 val')

            plt.xlabel('epoche')

            plt.ylabel('f1')

            plt.legend()

            plt.show()

            ansz=history.history['val_f1']

            mx=max(ansz)

            print('val f1 is maximal {} on a step {}'.format(mx,ansz.index(mx)+1))



  



    def predvec(self,x_test):

        return(self.model.predict(x_test))    

m1=Netn(embedding_matrix_Wiki)

m1.fit(0,600000,3,2000,loss='binary_crossentropy', metrics=['accuracy', f1],

              optimizer=Adam(lr=1e-3))

m1.fit(0,1250000,5,2000,loss='binary_crossentropy', metrics=['accuracy', f1],

              optimizer=Adam(lr=3e-4))

m1.unfreeze()

m1.fit(0,1250000,14,2000,loss='binary_crossentropy', metrics=['accuracy', f1],

              optimizer=Adam(lr=3e-5))
m2=Netn(embedding_matrix_Glove)
m2.fit(0,600000,3,2000,loss='binary_crossentropy', metrics=['accuracy', f1],

              optimizer=Adam(lr=1e-3))

m2.fit(0,1250000,5,2000,loss='binary_crossentropy', metrics=['accuracy', f1],

              optimizer=Adam(lr=3e-4))

m2.unfreeze()

m2.fit(0,1250000,14,2000,loss='binary_crossentropy', metrics=['accuracy', f1],

              optimizer=Adam(lr=3e-5))
lval=1250000

y_v=y_train.tolist()[lval:]

x_v=x_train[lval:]

u1=m1.predvec(x_v)

u2=m2.predvec(x_v)
def qf1(t,vv):

    tp=0

    fp=0

    fn=0

    yt=y_train.tolist()

    for i in range(len(vv)):

        if vv[-i]>t:

            if yt[-i]==1:

                tp+=1

            else:

                fp+=1

        else:

            if yt[-i]==1:

                fn+=1

    return (2*tp/(2*tp+fn+fp))



def best(vv):

    cc=0.35

    a=0

    for i in range(10):

        r= qf1(cc+0.01*i,vv)

        if (r>a):

            ii=i

            a=r

    a=0

    for i in range(10):

        r= qf1(cc+0.01*(ii-1)+0.002*i,vv)

        if (r>a):

            iii=i

            a=r

    

    print(a)

    return(cc+0.01*(ii-1)+0.002*iii)

    
bt=best((u1+u2)/2)

print(bt)
def ans(v,tr):

    res=np.zeros(len(v))

    for i in range(len(v)):

        if v[i]>tr:

            res[i]=1

    return res
v1=m1.predvec(x_test)

v2=m2.predvec(x_test)

vv=ans((v1+v2)/2,bt)
out = np.column_stack((test['qid'].values,vv))

np.savetxt('submission.csv', out, header="qid,prediction", 

            comments="", fmt="%s,%d")