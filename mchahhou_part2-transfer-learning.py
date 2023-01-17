from __future__ import division

import os

import random

import numpy as np

import tensorflow as tf

from keras import backend as K



import pandas as pd



from keras.layers import Input, Dropout, Dense, concatenate,  Embedding, Flatten, Activation, CuDNNLSTM,  Lambda

from keras.layers import Conv1D, Bidirectional, SpatialDropout1D, BatchNormalization, multiply

from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D

from keras import optimizers, callbacks, regularizers

from keras.models import Model





from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer



from sklearn.metrics import log_loss



import re



import gc

import time

import nltk



from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt



PATH = '../input/'

EMBEDDINGS_PATH = '../input/embeddings/'

WEIGHTS_PATH = './w0.h5'

MAX_TEXT_LENGTH = 40

EMBEDDING_SIZE  = 300





def embeddingNN(data, use_glove=True, trainable=True, seed=42):                                             

    np.random.seed(seed)



    emb_inpt = Input( shape=[data.shape[1]], name='emb_inpt')  

    if use_glove:

        x = Embedding(len( encoding_dc )+1, EMBEDDING_SIZE,

                      weights=[embedding_weights], trainable=trainable) (emb_inpt)      

    else:

        x = Embedding(len( encoding_dc )+1, EMBEDDING_SIZE) (emb_inpt)      

    

    x = CuDNNLSTM(64, return_sequences=False) (x)   

  

    x= Dense(1)(x)

    x = Activation('sigmoid')(x)

    

    model = Model([emb_inpt],x)



    return model





def run_model(lr=1e-3, bs=2048, use_glove=False, trainable=True):    

    predictions_test   = pd.DataFrame()

    predictions_train  = pd.DataFrame()

    for seed in range(3):

        es = callbacks.EarlyStopping( patience=2 )

        mc = callbacks.ModelCheckpoint( filepath=WEIGHTS_PATH, monitor='val_loss', mode='min', save_best_only=True )



        model = embeddingNN(X_test_emb, use_glove=use_glove, trainable=trainable, seed=seed)

        

        optimizer = optimizers.Adam(lr=lr)

        model.compile(loss="binary_crossentropy", optimizer=optimizer)



        model.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es, mc],

                     batch_size=bs, epochs=1000, verbose=2 )



        model.load_weights(WEIGHTS_PATH)



        p = model.predict(X_test_emb, batch_size=4096)

        predictions_test[str(seed)] = p.flatten()



        p = model.predict(X_train_emb, batch_size=4096)

        predictions_train[str(seed)] = p.flatten()



        print ( 'BAGGING SCORE Test: ' , log_loss(y_test,  predictions_test.mean(axis=1), eps = 1e-7) )

        print ( 'BAGGING SCORE Train: ', log_loss(y_train, predictions_train.mean(axis=1), eps = 1e-7) )

        
full_data = pd.read_csv(PATH+'train.csv',  encoding='utf-8', engine='python')

full_data['question_text'].fillna(u'unknownstring', inplace=True)



print (full_data.shape)
def preprocess( x ):

    x = re.sub( u"\s+", u" ", x ).strip()

    x = x.split(' ')[:MAX_TEXT_LENGTH]

    return ' '.join(x)





X_train, X_test, y_train, y_test = train_test_split(  full_data.question_text.values, full_data.target.values, 

                                                    shuffle =True, test_size=0.5, random_state=42)



X_train = np.array( [preprocess(x) for x in X_train] )

X_test  = np.array( [preprocess(x) for x in X_test] )



print ( X_train.shape, X_test.shape)
word_frequency_dc=defaultdict(np.uint32)

def word_count(text):

    text = text.split(' ')

    for w in text:

        word_frequency_dc[w]+=1



for x in X_train:

    word_count(x) 



encoding_dc = dict()

labelencoder=1

for key in word_frequency_dc:

    if word_frequency_dc[key]>1:

        encoding_dc[key]=labelencoder

        labelencoder+=1

    

print (len(encoding_dc))



def preprocess_keras(text):

    

    def get_encoding(w):

        if w in encoding_dc:

            return encoding_dc[w]

        return 0

    

    x = [ get_encoding(w) for w in text.split(' ') ]

    x = x + (MAX_TEXT_LENGTH-len(x))*[0]

    return x

X_train_emb = np.array( [ preprocess_keras(x) for x in X_train ] )

X_test_emb  = np.array( [ preprocess_keras(x) for x in X_test ]  )

print ( X_train_emb.shape, X_test_emb.shape)
#Function to load embeddings, returns a matrix (vovabulary size x embedding size)

def get_embeddings( word_index , method):    

    EMBEDDING_FILE = EMBEDDINGS_PATH+'glove.840B.300d/glove.840B.300d.txt'

    #each line of the file looks like : word dim1 dim2 .... dim300

    embeddings = { o.split(" ")[0]:np.asarray(o.split(" ")[1:], dtype='float32') for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index }

    

    temp = np.stack(embeddings.values())

    mean, std = temp.mean(), temp.std()

    print ('mean and std of GloVe weights :', mean, std)

    embedding_weights    = np.random.normal(mean, std, (len(word_index)+1,  EMBEDDING_SIZE ) ).astype(np.float32)



    for word, i in word_index.items():

        if (word in embeddings):

            embedding_weights[i] = embeddings.get(word)

    

    return embedding_weights



embedding_weights = get_embeddings(encoding_dc, method='glove')

print (embedding_weights.shape)
run_model( lr = 1e-2, bs=2048, use_glove=False )
run_model( lr = 5e-3, bs=2048, use_glove=True )