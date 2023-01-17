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
full_data = pd.read_csv(PATH+'train.csv',  encoding='utf-8', engine='python')

full_data['question_text'].fillna(u'unknownstring', inplace=True)



print (full_data.shape)
full_data.head(2)
sz = full_data['question_text'].apply(lambda x :len(x.split(' ')) )

_ = plt.hist(sz, bins=64)

plt.show()



MAX_TEXT_LENGTH=40
def preprocess( x ):

    x = re.sub( u"\s+", u" ", x ).strip()

    x = x.split(' ')[:MAX_TEXT_LENGTH]

    return ' '.join(x)





X_train, X_test, y_train, y_test = train_test_split(  full_data.question_text.values, full_data.target.values, 

                                                    shuffle =True, test_size=0.5, random_state=42)



X_train = np.array( [preprocess(x) for x in X_train] )

X_test  = np.array( [preprocess(x) for x in X_test] )



print ( X_train.shape, X_test.shape)
def my_tokenizer(x):

    return x.split(' ')



count_vectorizer = CountVectorizer( tokenizer=my_tokenizer, strip_accents = None,

                                   lowercase=False, dtype=np.float32,  min_df=2,

                                   ngram_range=(1, 1), stop_words=None, max_features=None)



X_train_bow = count_vectorizer.fit_transform(X_train)

X_test_bow  = count_vectorizer.transform(X_test)



print (X_train_bow.shape, X_test_bow.shape)

print ( len(count_vectorizer.vocabulary_ ) )
def bowNN(data, seed=42):                                             

    np.random.seed(seed)



    bow_inpt = Input( shape=[data.shape[1]], dtype = 'float32',   sparse = True, name='bow_inpt',)  

    

    x = Dense(100 )(bow_inpt)   

    x = Activation('relu')(x)

    

    x= Dense(1)(x)

    x = Activation('sigmoid')(x)

    

    model = Model([bow_inpt],x)



    return model



bowNN(X_test_bow).summary()



es = callbacks.EarlyStopping( patience=5 )



model_bow = bowNN(X_test_bow,seed=0)

model_bow.compile(loss="binary_crossentropy", optimizer=optimizers.Adam())



history = model_bow.fit(  X_train_bow, y_train, validation_data=(X_test_bow, y_test), callbacks=[es],

             batch_size=2048, epochs=1000 , verbose=2)



plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.legend( ['test', 'train'] )
preds = model_bow.predict( X_test_bow,batch_size=2048 )

print ( 'test score : ',log_loss(y_test, preds, eps = 1e-7) )


es = callbacks.EarlyStopping( patience=5 )

#below is the new callback to save the optimal weights

mc = callbacks.ModelCheckpoint( filepath=WEIGHTS_PATH, monitor='val_loss', mode='min', save_best_only=True )



model_bow = bowNN(X_test_bow,seed=0)

model_bow.compile(loss="binary_crossentropy", optimizer=optimizers.Adam())



history = model_bow.fit(  X_train_bow, y_train, validation_data=(X_test_bow, y_test), callbacks=[es, mc],

             batch_size=2048, epochs=1000 , verbose=2)

model_bow.load_weights( WEIGHTS_PATH )

preds = model_bow.predict( X_test_bow,batch_size=2048 )

print ( 'test score using the callback: ',log_loss(y_test, preds, eps = 1e-7) )

optimizers_list = [('sgd',optimizers.SGD( lr=.1) ),

                   ('sgd_momentum',optimizers.SGD(lr=.1, momentum=.9) ),

                   ('adagrad',optimizers.Adagrad()),

                   ('adadelta',optimizers.Adadelta()),

                   ('adam', optimizers.Adam()) 

                  ]



plt.figure(figsize=(20,5))



for optimizer in optimizers_list:

    es = callbacks.EarlyStopping( patience=5 )



    model_bow = bowNN(X_test_bow,seed=0)

    model_bow.compile(loss="binary_crossentropy", optimizer=optimizer[1])



    history = model_bow.fit(  X_train_bow, y_train, validation_data=(X_test_bow, y_test), callbacks=[es],

                 batch_size=2048, epochs=50, verbose=2 )

    

    

    plt.plot(history.history['val_loss'])

    

plt.legend([x[0] for x in optimizers_list], loc='upper right')

plt.title('model accuracy')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()
word_frequency_dc=defaultdict(np.uint32)

#function that computes the frequency of each word

def word_count(text):

    text = text.split(' ')

    for w in text:

        word_frequency_dc[w]+=1



for x in X_train:#!!!!!

    word_count(x) 

    

# we are using only the train dataset (X_train) to create the vocabulary

#we can't learn the embeddings of the words that are in Test but not in Train  anyway

#because they are not seen during the training phase

#For those unknown embeddings, we will, later, assign a single and shared embedding



#This second dict will store the label encodings for each word in the vocabulary we created

#Those encodings are needed as inputs for the NN

encoding_dc = dict() #Dont use defautdict here!!!!!



#We start the encoding at the value 1, we keep the value 0 for unknown words and padding

labelencoder=1

for key in word_frequency_dc:

    encoding_dc[key]=labelencoder

    labelencoder+=1

    

print ('vocabulary size : ', len(encoding_dc))
def preprocess_keras(text):  

    def get_encoding(w):

        if w in encoding_dc:       #We will assign a 0 to any word that is not in Train

            if encoding_dc[w]>1:   #For rare words (frequency==1), we return the 0 value

                return encoding_dc[w]

        return 0

    

    x = [ get_encoding(w) for w in text.split(' ') ]

    x = x + (MAX_TEXT_LENGTH-len(x))*[0] #we apply a padding so all questions have the same length, the padding is 0.

    return x



X_train_emb = np.array( [ preprocess_keras(x) for x in X_train ] )

X_test_emb  = np.array( [ preprocess_keras(x) for x in X_test ]  )

print ( X_train_emb.shape, X_test_emb.shape)
EMBEDDING_SIZE = 100

def embeddingNN(data, seed=42):                                             

    np.random.seed(seed)



    emb_inpt = Input( shape=[data.shape[1]], name='emb_inpt')  

    x = Embedding(len( encoding_dc )+1, EMBEDDING_SIZE) (emb_inpt)

    

    x = CuDNNLSTM(64, return_sequences=False) (x)   



    x= Dense(1)(x)

    x = Activation('sigmoid')(x)

    

    model = Model([emb_inpt],x)



    return model

embeddingNN(X_test_emb).summary()

es = callbacks.EarlyStopping( patience=3 )



model_emb = embeddingNN(X_test_emb,seed=0)

optimizer = optimizers.Adam(lr=1e-3)

model_emb.compile(loss="binary_crossentropy", optimizer=optimizer)



model_emb.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es],

             batch_size=2048, epochs=1000 , verbose=2)
encoding_dc = dict()

labelencoder=1

for key in word_frequency_dc:

    if word_frequency_dc[key]>1:# In the previous step, this condition was in the preprocess_keras function

        encoding_dc[key]=labelencoder

        labelencoder+=1

    

print ('vocabulary size : ', len(encoding_dc))



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
es = callbacks.EarlyStopping( patience=3 )



model_emb = embeddingNN(X_test_emb,seed=0)

optimizer = optimizers.Adam(lr=1e-3)

model_emb.compile(loss="binary_crossentropy", optimizer=optimizer)



model_emb.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es],

             batch_size=2048, epochs=1000 , verbose=2)
for EMBEDDING_SIZE in [50, 100, 300]:



    es = callbacks.EarlyStopping( patience=2 )



    model_emb = embeddingNN(X_test_emb,seed=0)

    optimizer = optimizers.Adam(lr=1e-3)

    model_emb.compile(loss="binary_crossentropy", optimizer=optimizer)



    model_emb.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es],

                 batch_size=2048, epochs=1000, verbose=2 )
EMBEDDING_SIZE = 100

for seed in range(3):

    es = callbacks.EarlyStopping( patience=2)



    model_emb = embeddingNN(X_test_emb,seed=seed)

    optimizer = optimizers.Adam(lr=1e-3)

    model_emb.compile(loss="binary_crossentropy", optimizer=optimizer)



    model_emb.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es],

                 batch_size=2048, epochs=1000 , verbose=2)
test_results = dict()

train_results = dict()

for EMBEDDING_SIZE in [50, 100, 300]:

    predictions_test   = pd.DataFrame()

    predictions_train  = pd.DataFrame()

    for seed in range(3):

        print ( 'Running Model with seed : ', seed, 'EMBEDDING_SIZE : ', EMBEDDING_SIZE )

        es = callbacks.EarlyStopping( patience=2, monitor='val_loss', mode='min' )

        mc = callbacks.ModelCheckpoint( filepath=WEIGHTS_PATH, monitor='val_loss', mode='min', save_best_only=True )



        model = embeddingNN(X_test_emb,seed=seed)# seed value change at each iteration

        optimizer = optimizers.Adam(lr=1e-3)

        model.compile(loss="binary_crossentropy", optimizer=optimizer)



        model.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es,mc],

                     batch_size=2048, epochs=1000, verbose=2 )



        model.load_weights(WEIGHTS_PATH)



        p = model.predict(X_test_emb, batch_size=4096)

        predictions_test[str(seed)] = p.flatten()

        

        p = model.predict(X_train_emb, batch_size=4096)

        predictions_train[str(seed)] = p.flatten()





    print('*'*50)



    test_results[EMBEDDING_SIZE]  =   log_loss(y_test, predictions_test.mean(axis=1), eps = 1e-7) 

    train_results[EMBEDDING_SIZE]  =   log_loss(y_train, predictions_train.mean(axis=1), eps = 1e-7) 

    

print ( test_results )
#We can see that a high dimensional (300) embedding gives the best results

{50: 0.11738275673258461, 100: 0.11665867249362337, 300: 0.11523046791034251}
run_this_code = False

if run_this_code:

    train_results = defaultdict(dict)

    test_results  = defaultdict(dict)



    for batch in [8192, 4096, 2048, 1024, 512]:

        for lr in [1e-4, 5e-4, 2e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 1e-1]:

            print ('                         Running Batch size : ', batch, 'Learning Rate : ', lr)

            predictions_train = pd.DataFrame()

            predictions_test  = pd.DataFrame()

            for seed in range(3):

                print ( 'Running Model with seed : ', seed )

                es = callbacks.EarlyStopping( patience=3, monitor='val_loss', mode='min' )

                mc = callbacks.ModelCheckpoint( filepath=WEIGHTS_PATH, monitor='val_loss', mode='min', save_best_only=True )



                model = embeddingNN(X_test_emb,seed=seed)

                optimizer = optimizers.Adam(lr=lr)

                model.compile(loss="binary_crossentropy", optimizer=optimizer)



                history = model.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es,mc],

                             batch_size=batch, epochs=1000, verbose=2 )



                model.load_weights(WEIGHTS_PATH)



                p = model.predict(X_test_emb, batch_size=1024)

                predictions_test[str(seed)] = p.flatten()



                p = model.predict(X_train_emb, batch_size=1024)

                predictions_train[str(seed)] = p.flatten()



                del model, history

                gc.collect()

                print ( 'BAGGING SCORE Test: ' , log_loss(y_test,  predictions_test.mean(axis=1), eps = 1e-7) )

                print ( 'BAGGING SCORE Train: ', log_loss(y_train, predictions_train.mean(axis=1), eps = 1e-7) )

            print('*'*50)

            print('')



            test_results[batch][lr]  = log_loss(y_test, predictions_test.mean(axis=1), eps = 1e-7) 

            train_results[batch][lr] = log_loss(y_train, predictions_train.mean(axis=1), eps = 1e-7) 
test_results = defaultdict(dict,

            {512: {0.0001: 0.11761565550762154,

              0.0002: 0.11698579357144163,

              0.0005: 0.11528072624713684,

              0.001: 0.11436596154065809,

              0.002: 0.1135644061933917,

              0.005: 0.11297268478063005,

              0.01: 0.11313634372237788,

              0.02: 0.11971199273251448,

              0.1: 0.21723223044694112},

             1024: {0.0001: 0.11825559436878119,

              0.0002: 0.11722233273331623,

              0.0005: 0.11577194832604475,

              0.001: 0.11503769556471356,

              0.002: 0.11409356554357286,

              0.005: 0.11365796929952472,

              0.01: 0.11303597386041803,

              0.02: 0.11768156400204072,

              0.1: 0.18597669868338992},

             2048: {0.0001: 0.11913019800933457,

              0.0002: 0.11773831646842034,

              0.0005: 0.11689393794967122,

              0.001: 0.11675256992165887,

              0.002: 0.11527158769441004,

              0.005: 0.11456169586485342,

              0.01: 0.11301434994527887,

              0.02: 0.1139797450196391,

              0.1: 0.16730059463991115},

             4096: {0.0001: 0.12076714592512715,

              0.0002: 0.1184375450744624,

              0.0005: 0.1182501223359833,

              0.001: 0.1181256172965128,

              0.002: 0.11741912951601145,

              0.005: 0.11483459988136224,

              0.01: 0.11376491009917047,

              0.02: 0.11450277734120927,

              0.1: 0.17871693992139087},

             8192: {0.0001: 0.1212227536500414,

              0.0002: 0.11957891040232248,

              0.0005: 0.11839599140357852,

              0.001: 0.11763529282024128,

              0.002: 0.11738116102242745,

              0.005: 0.11674775211161043,

              0.01: 0.11420364314091092,

              0.02: 0.11468364987433355,

              0.1: 0.151295587411758}})

pd.DataFrame(test_results)