import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import dill

import gensim

import pickle

import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,CuDNNLSTM, TimeDistributed

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

from sklearn.metrics import mean_squared_error

import sklearn

import gensim

import keras

from gensim.models.word2vec import Word2Vec

from gensim.models import KeyedVectors

import gc

import pickle

import gensim.downloader as api

import random

from sklearn.externals import joblib

from collections import defaultdict

import dill

import copy

import json

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import cross_validate

from math import sqrt

from sklearn.model_selection import KFold

import scipy



 

def rmsle(h, y):

    """

   Compute the Root Mean Squared Log Error for hypthesis h and targets y

   

   Args:

       h - numpy array containing predictions with shape (n_samples, n_targets)

       y - numpy array containing targets with shape (n_samples, n_targets)

   """

    return np.sqrt(np.square(np.log1p(h) - np.log1p(y)).mean())



def rmsle_v2(h, y):

    """

   Compute the Root Mean Squared Log Error for hypthesis h and targets y

   

   Args:

       h - numpy array containing predictions with shape (n_samples, n_targets)

       y - numpy array containing targets with shape (n_samples, n_targets)

   """

    return np.sqrt(np.square(h - y).mean())


############### Classification Version #######################



EMBEDDING_FILES = [

    #'../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]





NUM_MODELS = 1

BATCH_SIZE = 64

LSTM_UNITS = 200

DENSE_HIDDEN_UNITS = 2 * LSTM_UNITS

EPOCHS = 10

MAX_LEN = 100

CLASSES = 50



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, embedding, size):

    #embedding_index = load_embeddings(path)



    embedding_matrix = np.zeros((len(word_index) + 1, size))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding[word]

        except KeyError:

            pass

    return embedding_matrix

    



def build_model(embedding_matrix, one_hot_shape):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)    

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='concat')(x)

    #x = SpatialDropout1D(rate=0.3)(x)

    #x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='ave')(x)

    #x = SpatialDropout1D(rate=0.3)(x)



    #x = GlobalAveragePooling1D()(x) # this layer average each output from the Bidirectional layer 

    

    x = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])



    summary = Input(shape=(50,))

    x_aux = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)

    x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='concat')(x_aux)

    #x_aux = SpatialDropout1D(rate=0.3)(x_aux)

    #x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='ave')(x_aux)

    #x_aux = SpatialDropout1D(rate=0.3)(x_aux)

  

    #x_aux = GlobalAveragePooling1D()(x_aux)

    x_aux = concatenate([

        GlobalMaxPooling1D()(x_aux),

        GlobalAveragePooling1D()(x_aux),

    ])

    

    

    one_hot = Input(shape=(one_hot_shape,))

    hidden = concatenate([x,x_aux,one_hot])



    hidden = Dense(1000, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(800, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(500, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(400, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(300, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(100, activation='relu')(hidden)

    result = Dense(CLASSES, activation='linear')(hidden)

    result = Dense(CLASSES, activation='softmax')(result)

    

    model = Model(inputs=[words,summary,one_hot], outputs=[result])

    #adam = keras.optimizers.Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.5)

    #model.compile(loss='mse', optimizer='adam')

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    

    return model

    





#train total

train = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')

train = train[(train['num_votes'] <= 50)]

train['description'].fillna(' ',inplace=True)

#train = train[train['created_time'] > '2013-01-01 00:00:00']

#train = train.dropna(subset=['source'])

train.reset_index(inplace=True)

print(train.info())



#train reduced

#train = train[(train['num_votes'] > 1) & (train['num_votes'] < 50)]



#train baseline

train_baseline = train[~train['tag_type'].isna()]

train = train_baseline

train.reset_index(inplace=True)

#print(train.info())



train.loc[:,'description'] = train['description'].apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))

train.loc[:,'summary'] = train['summary'].apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))

train.loc[:,'latitude'] = train['latitude'].apply(lambda x: np.round(x,3))

train.loc[:,'longitude'] = train['longitude'].apply(lambda x: np.round(x,3))



train['hour'] = [str(pd.to_datetime(x).hour) for x in train['created_time']]

train['dayofweek'] = [str(pd.to_datetime(x).weekday()) for x in train['created_time']]

train['year'] = [str(pd.to_datetime(x).year) for x in train['created_time']]





tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train['description']) + list(train['summary']))



#test = train.sample(frac=.2,random_state=999)

#train =  train.iloc[~train.index.isin(test.index),:]

#y_train = train['num_votes']



y_train = np.zeros(shape=(len(train),CLASSES))

for index,sample in enumerate(train['num_votes']):    

    y_train[index,sample] = sample





description_train = train['description']

#description_test = test['description']

description_train = tokenizer.texts_to_sequences(description_train)

#description_test = tokenizer.texts_to_sequences(description_test)

description_train = sequence.pad_sequences(description_train, maxlen=MAX_LEN)

#description_test = sequence.pad_sequences(description_test, maxlen=MAX_LEN)



summary_train = train['summary']

#summary_test = test['summary']

summary_train = tokenizer.texts_to_sequences(summary_train)

#summary_test = tokenizer.texts_to_sequences(summary_test)

summary_train = sequence.pad_sequences(summary_train,50)

#summary_test = sequence.pad_sequences(summary_test,50)



from sklearn.preprocessing import OneHotEncoder

dummies = OneHotEncoder(handle_unknown='ignore')



#source_train = dummies.fit_transform(np.array(train[['source']]).reshape(-1,1)).todense()

#source = dummies.fit_transform(train[['source','num_votes']]) #this should be wrong but is issuing a smaller error

#source_test = dummies.transform(np.array(test[['source']]).reshape(-1,1)).todense()



# Decimal places      Object that can be unambiguously recognized at this scale

# 0	                  country or large region

# 1	            	  large city or district

# 2	             	  town or village

# 3               	  neighborhood, street

# 4                   individual street, land parcel

# 5                   individual trees, door entrance

# 6                   individual humans

# 7                   practical limit of commercial surveying

# 8                   specialized surveying (e.g. tectonic plate mapping)



latitude_train = train['latitude']

longitude_train = train['longitude']

#latitude_test = test['latitude']

#longitude_test = test['longitude']



onehot_train = np.stack((train['hour'],train['dayofweek'],train['year'],latitude_train,longitude_train),axis=-1)

onehot_train = dummies.fit_transform(onehot_train)

#onehot_train = scipy.sparse.csr_matrix(onehot_train)



#onehot_test = np.stack((test['hour'],test['dayofweek'],test['year'],latitude_test,longitude_test),axis=-1)

#onehot_test = dummies.transform(onehot_test)

#onehot_test = scipy.sparse.csr_matrix(onehot_test)





embedding = KeyedVectors.load_word2vec_format("../input/glove2word2vec/glove_w2v.txt",binary=False)

#embedding = KeyedVectors.load_word2vec_format("../input/glove840B300dtxt/glove.840B.300d.txt",binary=False)

EMBEDDINGS = [embedding]

embedding_matrix = build_matrix(tokenizer.word_index, embedding,200)

del embedding

gc.collect()

# embedding_matrix = np.concatenate(

#     [build_matrix(tokenizer.word_index, wordvect,200) for wordvect in EMBEDDINGS], axis=-1)





def main(x_train,y_train,one_hot_shape): 

  checkpoint_predictions = []

  weights = []



  model = build_model(embedding_matrix,one_hot_shape)



  for model_idx in range(NUM_MODELS):

        #model = build_model(embedding_matrix,one_hot_shape)

        for global_epoch in range(EPOCHS):

            model.fit(

                x_train,

                #[Y_train, y_aux_train],

                y_train,

                batch_size=BATCH_SIZE,

                epochs=1,

                verbose=2,

                #callbacks=[

                 #   LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

                #]

            )

            #checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

            #weights.append(2 ** global_epoch)



  model.save('lstm_model_3.h5')  

  return model



#This creates a matrix with all the train data. But is useless

#description_train = np.array(description_train).reshape(len(description_train),MAX_LEN)

#summary_train = np.array(summary_train).reshape(len(summary_train),50)

#final_train = scipy.sparse.hstack([scipy.sparse.csr_matrix(description_train),scipy.sparse.csr_matrix(summary_train),onehot_train],format='csr')



kfold = KFold(n_splits=2, shuffle=True, random_state=999)

cvscores = []

for train_idx,test_idx in kfold.split(onehot_train):    

 ## this is to guarantee full batches of BATCH_SIZE examples

 #length_train = len(train_idx)%BATCH_SIZE 

 #length_train = len(train_idx) - length_train

 #train_idx = train_idx[0:length_train]

 #test_idx = test_idx[0:length_train]    



 test_idx = (train.sample(frac=.2)).index

 train_idx = train[~train.index.isin(test_idx)].index



 model = main([description_train[train_idx], summary_train[train_idx], onehot_train[train_idx]], y_train[train_idx],onehot_train.get_shape()[1])

 pred = model.evaluate([description_train[test_idx],summary_train[test_idx],onehot_train[test_idx]], y_train[test_idx])

 #cvscores.append(np.mean(np.square(pred - y_train[test_idx])))

 #print('MSE %.3f' % np.mean(np.square(pred - y_train[test_idx])))

 print(model.metrics_names)

 print(pred)

 cvscores.append(pred[1])

 print('Accuracy %.3f' % pred[1])

              

print('Overall accuracy: %.3f' % np.mean(cvscores))





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# The below is necessary for starting Numpy generated random numbers

# in a well-defined initial state.

np.random.seed(999)

# The below is necessary for starting core Python generated random numbers

# in a well-defined state.

import random as rn

rn.seed(12345)



from tensorflow import set_random_seed

set_random_seed(2)



#train total

train = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')

train = train[(train['num_votes'] < 50)]

train['description'].fillna(' ',inplace=True)

train.reset_index(inplace=True)



#train reduced

#train = train[(train['num_votes'] > 1) & (train['num_votes'] < 50)]



#train baseline

train_baseline = train[~train['tag_type'].isna()]

#train = train_baseline

#train.reset_index(inplace=True)

#print(train.info())





train.loc[:,'description'] = train['description'].apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))

train.loc[:,'summary'] = train['summary'].apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))



EMBEDDING_FILES = [

    #'../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]





NUM_MODELS = 1

BATCH_SIZE = 64

LSTM_UNITS = 300

DENSE_HIDDEN_UNITS = 2 * LSTM_UNITS

EPOCHS = 10

MAX_LEN = 100



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, embedding, size):

    #embedding_index = load_embeddings(path)



    embedding_matrix = np.zeros((len(word_index) + 1, size))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding[word]

        except KeyError:

            pass

    return embedding_matrix

    



def build_model(embedding_matrix, num_aux_targets=0):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='concat')(x)

    #x = SpatialDropout1D(rate=0.3)(x)

    #x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='ave')(x)

    #x = SpatialDropout1D(rate=0.3)(x)



    #x = GlobalAveragePooling1D()(x) # this layer average each output from the Bidirectional layer 

    x = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

       

    summary = Input(shape=(50,))

    x_aux = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)

    x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='concat')(x_aux)

    #x_aux = SpatialDropout1D(rate=0.3)(x_aux)

    #x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='ave')(x_aux)

    #x_aux = SpatialDropout1D(rate=0.3)(x_aux)

    

    x_aux = GlobalAveragePooling1D()(x_aux)

    x_aux = concatenate([

        GlobalMaxPooling1D()(x_aux),

        GlobalAveragePooling1D()(x_aux),

    ])

  

    hidden = concatenate([x,x_aux])   



    hidden = Dense(1000,activation='relu')(hidden)

    hidden = Dropout(0.5)(hidden)

    hidden = Dense(900, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(800, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(700, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(500, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)    

    hidden = Dense(200, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)    

    hidden = Dense(100, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)    

 

    result = Dense(1, activation='linear')(hidden)

    model = Model(inputs=[words,summary], outputs=[result])

    #adam = Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.5)

    model.compile(loss='mse', optimizer='adam')



    return model

    



def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data





tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train['description']) + list(train['summary']))



#test = train.sample(frac=.2,random_state=999)

#train =  train.iloc[~train.index.isin(test.index),:]



#print(test.index.isin([194239,21704,3014,24985,17063,223071,19349,26392,31870,189610,37705]))



description_train = train['description']

#description_test = test['description']

description_train = tokenizer.texts_to_sequences(description_train)

#description_test = tokenizer.texts_to_sequences(description_test)

description_train = sequence.pad_sequences(description_train, maxlen=MAX_LEN)

#description_test = sequence.pad_sequences(description_test, maxlen=MAX_LEN)



summary_train = train['summary']

#summary_test = test['summary']

summary_train = tokenizer.texts_to_sequences(summary_train)

#summary_test = tokenizer.texts_to_sequences(summary_test)

summary_train = sequence.pad_sequences(summary_train,50)

#summary_test = sequence.pad_sequences(summary_test,50)



#y_test = test['num_votes'].apply(lambda x: np.log1p(x))

#y_train = train['num_votes'].apply(lambda x: np.log1p(x))



#y_test = test['num_votes']

y_train = train['num_votes']





#embedding = KeyedVectors.load_word2vec_format("../input/glove2word2vec/glove_w2v.txt",binary=False)

#embedding = KeyedVectors.load_word2vec_format("glove300d_word2vec.txt",binary=False)

embedding = KeyedVectors.load_word2vec_format("../input/word2vec-google/GoogleNews-vectors-negative300.bin",binary=True)



EMBEDDINGS = [embedding]

embedding_matrix = build_matrix(tokenizer.word_index, embedding,300)

del embedding

gc.collect()

# embedding_matrix = np.concatenate(

#     [build_matrix(tokenizer.word_index, wordvect,200) for wordvect in EMBEDDINGS], axis=-1)



def main(x_train,y_train): 

  checkpoint_predictions = []

  weights = []



  model = build_model(embedding_matrix)



  for model_idx in range(NUM_MODELS):

        #model = build_model(embedding_matrix)

        for global_epoch in range(EPOCHS):

            model.fit(

                x_train,

                #[Y_train, y_aux_train],

                y_train,

                batch_size=BATCH_SIZE,

                epochs=1,

                verbose=2,

                callbacks=[

                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

                ]

            )

            #checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

            #weights.append(2 ** global_epoch)



  return model



kfold = KFold(n_splits=2, shuffle=True, random_state=999)

cvscores = []

for train_idx, test_idx in kfold.split(description_train):    

 test_idx = (train.sample(frac=.2)).index

 train_idx = train[~train.index.isin(test_idx)].index



 model = main([description_train[train_idx],summary_train[train_idx]],y_train[train_idx])

 pred = model.predict([description_train[test_idx],summary_train[test_idx]]).flatten()

 cvscores.append(np.mean(np.square(pred - y_train[test_idx])))

 print('MSE %.3f' % np.mean(np.square(pred - y_train[test_idx])))

              

print('Overall MSE: %.3f' % np.mean(cvscores))



                 

#from sklearn.metrics import r2_score

#print("R-squared: ",end="")

#print(r2_score(y_test, pred))



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







EMBEDDING_FILES = [

    #'../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]





NUM_MODELS = 1

BATCH_SIZE = 64

LSTM_UNITS = 200

DENSE_HIDDEN_UNITS = 2 * LSTM_UNITS

EPOCHS = 10

MAX_LEN = 100



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, embedding, size):

    #embedding_index = load_embeddings(path)



    embedding_matrix = np.zeros((len(word_index) + 1, size))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding[word]

        except KeyError:

            pass

    return embedding_matrix

    



def build_model(embedding_matrix, one_hot_shape):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)    

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='concat')(x)

    #x = SpatialDropout1D(rate=0.3)(x)

    #x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='ave')(x)

    #x = SpatialDropout1D(rate=0.3)(x)



    #x = GlobalAveragePooling1D()(x) # this layer average each output from the Bidirectional layer 

    

    x = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])



    summary = Input(shape=(50,))

    x_aux = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)

    x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='concat')(x_aux)

    #x_aux = SpatialDropout1D(rate=0.3)(x_aux)

    #x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True),merge_mode='ave')(x_aux)

    #x_aux = SpatialDropout1D(rate=0.3)(x_aux)

  

    #x_aux = GlobalAveragePooling1D()(x_aux)

    x_aux = concatenate([

        GlobalMaxPooling1D()(x_aux),

        GlobalAveragePooling1D()(x_aux),

    ])

    

    

    one_hot = Input(shape=(one_hot_shape,))

    hidden = concatenate([x,x_aux,one_hot])



    hidden = Dense(1000, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(800, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(500, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(400, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(300, activation='relu')(hidden)

    hidden = Dropout(0.4)(hidden)

    hidden = Dense(100, activation='relu')(hidden)

    result = Dense(1, activation='linear')(hidden)

    

    model = Model(inputs=[words,summary,one_hot], outputs=[result])

    #adam = keras.optimizers.Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.5)

    model.compile(loss='mse', optimizer='adam')



    return model

    



def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data





#train total

train = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')

train = train[(train['num_votes'] < 50)]

train['description'].fillna(' ',inplace=True)

#train = train[train['created_time'] > '2013-01-01 00:00:00']

#train = train.dropna(subset=['source'])

train.reset_index(inplace=True)

print(train.info())



#train reduced

#train = train[(train['num_votes'] > 1) & (train['num_votes'] < 50)]



#train baseline

train_baseline = train[~train['tag_type'].isna()]

train = train_baseline

train.reset_index(inplace=True)

#print(train.info())



train.loc[:,'description'] = train['description'].apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))

train.loc[:,'summary'] = train['summary'].apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))

train.loc[:,'latitude'] = train['latitude'].apply(lambda x: np.round(x,3))

train.loc[:,'longitude'] = train['longitude'].apply(lambda x: np.round(x,3))



train['hour'] = [str(pd.to_datetime(x).hour) for x in train['created_time']]

train['dayofweek'] = [str(pd.to_datetime(x).weekday()) for x in train['created_time']]

train['year'] = [str(pd.to_datetime(x).year) for x in train['created_time']]





tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train['description']) + list(train['summary']))



#test = train.sample(frac=.2,random_state=999)

#train =  train.iloc[~train.index.isin(test.index),:]



#y_test = test['num_votes'].apply(lambda x: np.log1p(x+1))

#y_train = train['num_votes'].apply(lambda x: np.log1p(x+1))

#y_test = test['num_votes']

y_train = train['num_votes']





description_train = train['description']

#description_test = test['description']

description_train = tokenizer.texts_to_sequences(description_train)

#description_test = tokenizer.texts_to_sequences(description_test)

description_train = sequence.pad_sequences(description_train, maxlen=MAX_LEN)

#description_test = sequence.pad_sequences(description_test, maxlen=MAX_LEN)



summary_train = train['summary']

#summary_test = test['summary']

summary_train = tokenizer.texts_to_sequences(summary_train)

#summary_test = tokenizer.texts_to_sequences(summary_test)

summary_train = sequence.pad_sequences(summary_train,50)

#summary_test = sequence.pad_sequences(summary_test,50)



from sklearn.preprocessing import OneHotEncoder

dummies = OneHotEncoder(handle_unknown='ignore')



#source_train = dummies.fit_transform(np.array(train[['source']]).reshape(-1,1)).todense()

#source = dummies.fit_transform(train[['source','num_votes']]) #this should be wrong but is issuing a smaller error

#source_test = dummies.transform(np.array(test[['source']]).reshape(-1,1)).todense()



# Decimal places      Object that can be unambiguously recognized at this scale

# 0	                  country or large region

# 1	            	  large city or district

# 2	             	  town or village

# 3               	  neighborhood, street

# 4                   individual street, land parcel

# 5                   individual trees, door entrance

# 6                   individual humans

# 7                   practical limit of commercial surveying

# 8                   specialized surveying (e.g. tectonic plate mapping)



latitude_train = train['latitude']

longitude_train = train['longitude']

#latitude_test = test['latitude']

#longitude_test = test['longitude']



onehot_train = np.stack((train['hour'],train['dayofweek'],train['year'],latitude_train,longitude_train),axis=-1)

onehot_train = dummies.fit_transform(onehot_train)

#onehot_train = scipy.sparse.csr_matrix(onehot_train)



#onehot_test = np.stack((test['hour'],test['dayofweek'],test['year'],latitude_test,longitude_test),axis=-1)

#onehot_test = dummies.transform(onehot_test)

#onehot_test = scipy.sparse.csr_matrix(onehot_test)





embedding = KeyedVectors.load_word2vec_format("../input/glove2word2vec/glove_w2v.txt",binary=False)

#embedding = KeyedVectors.load_word2vec_format("../input/glove840B300dtxt/glove.840B.300d.txt",binary=False)

EMBEDDINGS = [embedding]

embedding_matrix = build_matrix(tokenizer.word_index, embedding,200)

del embedding

gc.collect()

# embedding_matrix = np.concatenate(

#     [build_matrix(tokenizer.word_index, wordvect,200) for wordvect in EMBEDDINGS], axis=-1)





def main(x_train,y_train,one_hot_shape): 

  checkpoint_predictions = []

  weights = []



  model = build_model(embedding_matrix,one_hot_shape)



  for model_idx in range(NUM_MODELS):

        #model = build_model(embedding_matrix,one_hot_shape)

        for global_epoch in range(EPOCHS):

            model.fit(

                x_train,

                #[Y_train, y_aux_train],

                y_train,

                batch_size=BATCH_SIZE,

                epochs=1,

                verbose=2,

                #callbacks=[

                 #   LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

                #]

            )

            #checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

            #weights.append(2 ** global_epoch)



  model.save('lstm_model_3.h5')  

  return model



#This creates a matrix with all the train data. But is useless

#description_train = np.array(description_train).reshape(len(description_train),MAX_LEN)

#summary_train = np.array(summary_train).reshape(len(summary_train),50)

#final_train = scipy.sparse.hstack([scipy.sparse.csr_matrix(description_train),scipy.sparse.csr_matrix(summary_train),onehot_train],format='csr')



kfold = KFold(n_splits=2, shuffle=True, random_state=999)

cvscores = []

for train_idx,test_idx in kfold.split(onehot_train):    

 ## this is to guarantee full batches of BATCH_SIZE examples

 #length_train = len(train_idx)%BATCH_SIZE 

 #length_train = len(train_idx) - length_train

 #train_idx = train_idx[0:length_train]

 #test_idx = test_idx[0:length_train]    



 test_idx = (train.sample(frac=.2)).index

 train_idx = train[~train.index.isin(test_idx)].index



 model = main([description_train[train_idx], summary_train[train_idx], onehot_train[train_idx]], y_train[train_idx],onehot_train.get_shape()[1])

 pred = model.predict([description_train[test_idx],summary_train[test_idx],onehot_train[test_idx]]).flatten()

 cvscores.append(np.mean(np.square(pred - y_train[test_idx])))

 print('MSE %.3f' % np.mean(np.square(pred - y_train[test_idx])))

              

print('Overall MSE: %.3f' % np.mean(cvscores))





indexes = list(range(0,len(description_train)))

print(len(indexes))

print((train.index))

test_idx = random.sample(indexes,k=int(0.2*len(indexes)))



for v in test_idx:

    indexes.remove(v)

print(len(indexes))

print(len(test_idx))

TRAIN = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')



import matplotlib.pyplot as plt

import matplotlib.ticker as plticker



loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals

#Number of non null entries for each column

TRAIN.info()



TRAIN = TRAIN[(TRAIN['num_votes'] < 50)]

TRAIN['description'].fillna(' ',inplace=True)

TRAIN.reset_index(inplace=True)



TRAIN.loc[:,'latitude'] = TRAIN['latitude'].apply(lambda x: np.round(x,3))

TRAIN.loc[:,'longitude'] = TRAIN['longitude'].apply(lambda x: np.round(x,3))



TRAIN['hour'] = [str(pd.to_datetime(x).hour) for x in TRAIN['created_time']]

TRAIN['dayofweek'] = [str(pd.to_datetime(x).weekday()) for x in TRAIN['created_time']]

TRAIN['year'] = [str(pd.to_datetime(x).year) for x in TRAIN['created_time']]





# frequency of votes

count_votes = pd.DataFrame(TRAIN['num_votes'].value_counts(normalize=True)) #counts the number of times each type occured

ax = count_votes.plot(kind='bar',figsize=(10,5), title="Frequency of votes")

ax.set_ylabel('% of votes')

#ax.set_xlabel('Models')



# frequency of votes baseline

baseline = TRAIN[~TRAIN['tag_type'].isna()]

count_votes = pd.DataFrame(baseline['num_votes'].value_counts(normalize=True)) #counts the number of times each type occured

ax = count_votes.plot(kind='bar',figsize=(10,5), title="Frequency of votes baseline")

ax.set_ylabel('% of votes')

#ax.set_xlabel('Models')

# Total number of examples by issue type for baseline

count_examples = baseline[['tag_type','num_votes']].groupby(['tag_type']).count()

ax = count_examples.plot(kind='bar',figsize=(10,5), title="Number of examples")

ax.set_xlabel('Issue type')





# Frequency issue type

count_issues = pd.DataFrame(TRAIN['tag_type'].value_counts(normalize=True)) #counts the number of times each type occured

ax = count_issues.plot(kind='bar',figsize=(10,5), title="Frequency of issue type")

ax.set_ylabel('% of issues')

#ax.set_xlabel('Models')



#Average Number of votes per issue

votes_per_issue_type = TRAIN.loc[:,['tag_type','num_votes']].groupby('tag_type').mean()

votes_per_issue_type.sort_values(by=['num_votes'],ascending=False,inplace=True)

ax = votes_per_issue_type.plot(kind='bar',figsize=(10,5), title="Average number of votes per issue type")

ax.set_ylabel('Average number of votes')



#Variance of num_votes

print("Mean of num votes all dataset: %f " % np.mean(TRAIN['num_votes']))

print("Variance of num votes all dataset: %f " % np.var(TRAIN['num_votes']))

aux = TRAIN[(TRAIN['num_votes']>1)&(TRAIN['num_votes']<50)]

print("Mean of num votes reduced dataset: %f " % np.mean(aux['num_votes']))

print("Variance of num votes reduced dataset: %f " % np.var(aux['num_votes']))

aux = TRAIN[~TRAIN['tag_type'].isna()]

print("Mean of num votes baseline dataset: %f " % np.mean(aux['num_votes']))

print("Variance of num votes baseline dataset: %f " % np.var(aux['num_votes']))



#Variance per type

variance_per_issue_type = TRAIN.loc[:,['tag_type','num_votes']].groupby('tag_type').var()

variance_per_issue_type.sort_values(by=['num_votes'],ascending=False,inplace=True)

ax = variance_per_issue_type.plot(kind='bar',figsize=(10,5), title="Variance of votes per issue type")

ax.set_ylabel('Variance of votes')



#Find sensitive exemples to test 

#Get max and min votes for Traffic

print(aux.loc[aux['tag_type'] == 'traffic',['id','num_votes']].sort_values(by=['num_votes']))

#194239  301444         17

#21704   104882         18

#3014    187575         19

#24985   312502         19

#17063    17969         25

#223071   88840         26

#19349   203600         31

#26392   296264         34

#31870   233309         35

#189610  173886         71

#37705   243191        134

print(aux.iloc[aux.index.isin([194239,21704,3014,24985,17063,223071,19349,26392,31870,189610,37705])])



#Average Number of votes per issue for Okland

print('Okland')

okland = TRAIN[(TRAIN['latitude'] >= 37.80) & (TRAIN['latitude'] <= 38)]

votes_per_issue_type = okland.loc[:,['tag_type','num_votes']].groupby('tag_type').mean()

votes_per_issue_type.sort_values(by=['num_votes'],ascending=False,inplace=True)

#print(votes_per_issue_type)

ax = votes_per_issue_type.plot(kind='bar',figsize=(10,5), title="Average number of votes per issue type for Okland")

ax.set_ylabel('Average number of votes')







# Ratio number_of_votes:count_issue_type - number of votes per issue type normalized by the count of each issue type

#for i,sample in count_issues.iterrows():

#    votes_per_issue_type.loc[i,'num_votes'] /= count_issues.loc[i,'tag_type']

#votes_per_issue_type.sort_values(by=['num_votes'],ascending=False,inplace=True)

#ax = votes_per_issue_type.plot(kind='bar',figsize=(10,5), title="")

#ax.set_ylabel('Ratio number_votes:number_occurences')





#Total number per issue type for null descriptions

#index_non_null = TRAIN[TRAIN['description'].notnull()].index

#count_issues = pd.DataFrame(TRAIN.loc[~TRAIN.index.isin(index_non_null),'tag_type'].value_counts())

#ax = count_issues.plot(kind='bar',figsize=(10,5), title="Number of examples where description is null")

#ax.set_ylabel('Total number of issues')

#ax.set_xlabel('Models')



#Comparing number of votes by issue type among different neighborhoods 

locations_more_than_one_issue = TRAIN.loc[:,['latitude','longitude','num_votes']].groupby(['latitude','longitude']).count()

locations_more_than_one_issue.sort_values(by=['num_votes'],ascending=False,inplace=True)

for v in locations_more_than_one_issue.iterrows():

    #print(v[0][0])

    if v[1][0] < 30:

        break

        

    if all(TRAIN.loc[(TRAIN['latitude'] == v[0][0])&(TRAIN['longitude'] == v[0][1]),'tag_type'].isna()):

        continue

        

    aux = TRAIN.loc[(TRAIN['latitude'] == v[0][0])&(TRAIN['longitude'] == v[0][1]),

                    ['tag_type','num_votes']].groupby(['tag_type']).mean()

    ax = aux.plot(kind='barh',figsize=(10,5), title=str(v[0][0]) + ' ' + str(v[0][1]))

    for i, v in enumerate(aux.values):

        #plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold') 

        ax.text(v, i, str(np.round(v,2)), color='blue', fontweight='bold')

    plt.show()



    



#This prints the ten max locations and ten min locations per tag_type 

#votes_per_location_issue_type = TRAIN.loc[TRAIN['location'].isin(locations_more_than_one_issue.index),['tag_type','location','num_votes']].groupby(['location','tag_type']).sum()

#votes_per_location_issue_type.sort_index(by=['location'],inplace=True)

#potholes = votes_per_location_issue_type[votes_per_location_issue_type.index.isin(['pothole'],level=1)]

#potholes_max = potholes.sort_values(by=['num_votes'],ascending=False)

#potholes_min = potholes.sort_values(by=['num_votes'],ascending=True)

#ax = pd.concat([potholes_max.iloc[0:10,:],potholes_min.iloc[0:10,:]]).plot(kind='bar',figsize=(10,5), title="")

#tress = votes_per_location_issue_type[votes_per_location_issue_type.index.isin(['tree'],level=1)]

#tress_max = tress.sort_values(by=['num_votes'],ascending=False)

#tress_min = tress.sort_values(by=['num_votes'],ascending=True)

#ax = pd.concat([tress_max.iloc[0:10,:],tress_min.iloc[0:10,:]]).plot(kind='bar',figsize=(10,5), title="")





    

# Average number of words

index_non_null = TRAIN[TRAIN['description'].notnull()].index

max = 0

min = np.inf

avg = 0

for i,w in TRAIN.iloc[index_non_null].iterrows():

    words = w['description'].split()

    if len(words) > max:

        max = len(words)

    if len(words) < min:

        min = len(words)

    avg += len(words)

avg /= len(index_non_null)



print('max: %d' %max)

print('min: %d' %min)

print('avg: %f' %avg)





index_non_null = TRAIN[TRAIN['summary'].notnull()].index

max = 0

min = np.inf

avg = 0

for i,w in TRAIN.iloc[index_non_null].iterrows():

    words = w['summary'].split()

    if len(words) > max:

        max = len(words)

    if len(words) < min:

        min = len(words)

    avg += len(words)

avg /= len(index_non_null)



print('max: %d' %max)

print('min: %d' %min)

print('avg: %f' %avg)
TRAIN = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')

from datetime import date

from datetime import datetime

from datetime import time



#Number of non null entries for each column

TRAIN.info()



print(TRAIN[~TRAIN['tag_type'].isna()].info())

print(TRAIN[TRAIN['tag_type'].notnull()].info())



print((date.today()))

print(date.today().strftime('%Y-%m-%d'))

#datetime.datetime.strptime(today_date,'%m/%d/%y')



#len(TRAIN[TRAIN['tag_type'].notnull()])datetime.date.today() datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

TRAIN['created_time'] = TRAIN.loc[:,'created_time'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))

print(TRAIN[['created_time','num_votes']].groupby(['created_time']).count())

print(np.mean(TRAIN[['created_time','num_votes']].groupby(['created_time']).count()['num_votes']))
import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.graphics.gofplots import ProbPlot



plt.style.use('seaborn') # pretty matplotlib plots

plt.rc('font', size=14)

plt.rc('figure', titlesize=18)

plt.rc('axes', labelsize=15)

plt.rc('axes', titlesize=18)



def graph(formula, x_range, label=None):

    """

    Helper function for plotting cook's distance lines

    """

    x = x_range

    y = formula(x)

    plt.plot(x, y, label=label, lw=1, ls='--', color='red')





def diagnostic_plots(X, y, model_fit=None):

  """

  Function to reproduce the 4 base plots of an OLS model in R.



  ---

  Inputs:



  X: A numpy array or pandas dataframe of the features to use in building the linear regression model



  y: A numpy array or pandas series/dataframe of the target variable of the linear regression model



  model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. If not provided, will be

                        generated from X, y

  """



  if not model_fit:

      model_fit = sm.OLS(y, sm.add_constant(X)).fit()



  print(model_fit.summary())

  # create dataframe from X, y for easier plot handling

  #dataframe = pd.concat([X, y], axis=1)



  # model values

  model_fitted_y = model_fit.fittedvalues

  # model residuals

  model_residuals = model_fit.resid

  # normalized residuals

  model_norm_residuals = model_fit.get_influence().resid_studentized_internal

  # absolute squared normalized residuals

  model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

  # absolute residuals

  model_abs_resid = np.abs(model_residuals)

  # leverage, from statsmodels internals

  model_leverage = model_fit.get_influence().hat_matrix_diag

  # cook's distance, from statsmodels internals

  model_cooks = model_fit.get_influence().cooks_distance[0]



  plot_lm_1 = plt.figure()

  plot_lm_1.axes[0] = sns.residplot(model_fitted_y, y, data=None,

                            lowess=True,

                            scatter_kws={'alpha': 0.5},

                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



  plot_lm_1.axes[0].set_title('Residuals vs Fitted')

  plot_lm_1.axes[0].set_xlabel('Fitted values')

  plot_lm_1.axes[0].set_ylabel('Residuals');



  # annotations

  abs_resid = model_abs_resid.sort_values(ascending=False)

  abs_resid_top_3 = abs_resid[:3]

  for i in abs_resid_top_3.index:

      plot_lm_1.axes[0].annotate(i,

                                 xy=(model_fitted_y[i],

                                     model_residuals[i]));



  QQ = ProbPlot(model_norm_residuals)

  plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

  plot_lm_2.axes[0].set_title('Normal Q-Q')

  plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')

  plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

  # annotations

  abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)

  abs_norm_resid_top_3 = abs_norm_resid[:3]

  for r, i in enumerate(abs_norm_resid_top_3):

      plot_lm_2.axes[0].annotate(i,

                                 xy=(np.flip(QQ.theoretical_quantiles, 0)[r],

                                     model_norm_residuals[i]));



  plot_lm_3 = plt.figure()

  plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);

  sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,

              scatter=False,

              ci=False,

              lowess=True,

              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});

  plot_lm_3.axes[0].set_title('Scale-Location')

  plot_lm_3.axes[0].set_xlabel('Fitted values')

  plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');



  # annotations

  abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)

  abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

  for i in abs_norm_resid_top_3:

      try:

       plot_lm_3.axes[0].annotate(i,

                                 xy=(model_fitted_y[i],

                                     model_norm_residuals_abs_sqrt[i]));

      except:

          pass



  plot_lm_4 = plt.figure();

  plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);

  sns.regplot(model_leverage, model_norm_residuals,

              scatter=False,

              ci=False,

              lowess=True,

              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});

  plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)

  plot_lm_4.axes[0].set_ylim(-3, 5)

  plot_lm_4.axes[0].set_title('Residuals vs Leverage')

  plot_lm_4.axes[0].set_xlabel('Leverage')

  plot_lm_4.axes[0].set_ylabel('Standardized Residuals');



  # annotations

  leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

  for i in leverage_top_3:

      plot_lm_4.axes[0].annotate(i,

                                 xy=(model_leverage[i],

                                     model_norm_residuals[i]));



  p = len(model_fit.params) # number of model parameters

  graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),

        np.linspace(0.001, max(model_leverage), 50),

        'Cook\'s distance') # 0.5 line

  graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),

        np.linspace(0.001, max(model_leverage), 50)) # 1 line

  plot_lm_4.legend(loc='upper right');





train = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')

train = train[(train['num_votes'] < 50)]

train['description'].fillna(' ',inplace=True)

#train = train[~train['tag_type'].isna()]

train.reset_index(inplace=True)



train.loc[:,'latitude'] = train['latitude'].apply(lambda x: np.round(x,3))

train.loc[:,'longitude'] = train['longitude'].apply(lambda x: np.round(x,3))



train['hour'] = [str(pd.to_datetime(x).hour) for x in train['created_time']]

train['dayofweek'] = [str(pd.to_datetime(x).weekday()) for x in train['created_time']]

train['year'] = [str(pd.to_datetime(x).year) for x in train['created_time']]



from sklearn.preprocessing import OneHotEncoder

dummies = OneHotEncoder(categories='auto')



latitude_train = train['latitude']

longitude_train = train['longitude']



onehot_train = np.stack((train['hour'],train['dayofweek'],train['year'],latitude_train,longitude_train),axis=-1)

onehot_train = dummies.fit_transform(onehot_train)



text_fields = onehot_train





############################### StatsModels ###################################

######### Generating Diagnostic Plots for the data #################



#y_train = train['num_votes'].apply(lambda x: np.log(x))

y_train = train['num_votes']

#

rows = int(.5 * (text_fields.get_shape()[0]))

text_fields = text_fields[0:rows,:]

text_fields = text_fields.toarray()

diagnostic_plots(text_fields,y_train[0:rows])

plt.show()


