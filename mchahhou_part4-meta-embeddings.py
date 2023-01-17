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

import operator 



import matplotlib.pyplot as plt



PATH = '../input/'

EMBEDDINGS_PATH = '../input/embeddings/'

WEIGHTS_PATH = './w0.h5'

MAX_TEXT_LENGTH = 40

EMBEDDING_SIZE  = 300









def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in vocab:

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))







contraction_mapping = {u"ain't": u"is not", u"aren't": u"are not",u"can't": u"cannot", u"'cause": u"because",

                       u"could've": u"could have", u"couldn't": u"could not", u"didn't": u"did not",

                       u"doesn't": u"does not", u"don't": u"do not", u"hadn't": u"had not",

                       u"hasn't": u"has not", u"haven't": u"have not", u"he'd": u"he would",

                       u"he'll": u"he will", u"he's": u"he is", u"how'd": u"how did", u"how'd'y": u"how do you",

                       u"how'll": u"how will", u"how's": u"how is",  u"I'd": u"I would",

                       u"I'd've": u"I would have", u"I'll": u"I will", u"I'll've": u"I will have",

                       u"I'm": u"I am", u"I've": u"I have", u"i'd": u"i would", u"i'd've": u"i would have",

                       u"i'll": u"i will",  u"i'll've": u"i will have",u"i'm": u"i am", u"i've": u"i have",

                       u"isn't": u"is not", u"it'd": u"it would", u"it'd've": u"it would have",

                       u"it'll": u"it will", u"it'll've": u"it will have",u"it's": u"it is",

                       u"let's": u"let us", u"ma'am": u"madam", u"mayn't": u"may not",

                       u"might've": u"might have",u"mightn't": u"might not",u"mightn't've": u"might not have",

                       u"must've": u"must have", u"mustn't": u"must not", u"mustn't've": u"must not have",

                       u"needn't": u"need not", u"needn't've": u"need not have",u"o'clock": u"of the clock",

                       u"oughtn't": u"ought not", u"oughtn't've": u"ought not have", u"shan't": u"shall not", 

                       u"sha'n't": u"shall not", u"shan't've": u"shall not have", u"she'd": u"she would",

                       u"she'd've": u"she would have", u"she'll": u"she will", u"she'll've": u"she will have",

                       u"she's": u"she is", u"should've": u"should have", u"shouldn't": u"should not",

                       u"shouldn't've": u"should not have", u"so've": u"so have",u"so's": u"so as",

                       u"this's": u"this is",u"that'd": u"that would", u"that'd've": u"that would have",

                       u"that's": u"that is", u"there'd": u"there would", u"there'd've": u"there would have",

                       u"there's": u"there is", u"here's": u"here is",u"they'd": u"they would", 

                       u"they'd've": u"they would have", u"they'll": u"they will", 

                       u"they'll've": u"they will have", u"they're": u"they are", u"they've": u"they have", 

                       u"to've": u"to have", u"wasn't": u"was not", u"we'd": u"we would",

                       u"we'd've": u"we would have", u"we'll": u"we will", u"we'll've": u"we will have", 

                       u"we're": u"we are", u"we've": u"we have", u"weren't": u"were not",

                       u"what'll": u"what will", u"what'll've": u"what will have", u"what're": u"what are",

                       u"what's": u"what is", u"what've": u"what have", u"when's": u"when is",

                       u"when've": u"when have", u"where'd": u"where did", u"where's": u"where is",

                       u"where've": u"where have", u"who'll": u"who will", u"who'll've": u"who will have",

                       u"who's": u"who is", u"who've": u"who have", u"why's": u"why is", u"why've": u"why have",

                       u"will've": u"will have", u"won't": u"will not", u"won't've": u"will not have",

                       u"would've": u"would have", u"wouldn't": u"would not", u"wouldn't've": u"would not have",

                       u"y'all": u"you all", u"y'all'd": u"you all would",u"y'all'd've": u"you all would have",

                       u"y'all're": u"you all are",u"y'all've": u"you all have",u"you'd": u"you would",

                       u"you'd've": u"you would have", u"you'll": u"you will", u"you'll've": u"you will have",

                       u"you're": u"you are", u"you've": u"you have", u"didnt": u"did not" }



def remove_special_chars(w):

    for i, j in [ (u"é", u"e"), (u"ē", u"e"), (u"è", u"e"), (u"ê", u"e"), (u"à", u"a"),

                 (u"â", u"a"), (u"ô", u"o"), (u"ō", u"o"), (u"ü", u"u"), (u"ï", u"i"),

                 (u"ç", u"c"), (u"\xed", u"i")]:

        x = re.sub(i, j, w)

        if x in embeddings_index:

            return x

        

    return w



def lower(w):

    x = w.lower()

    if x in embeddings_index:

        return x

    else:

        return w

    

def keep_alpha_num(w):

    x = re.sub(u"[^a-z\s0-9]", u" ", w)

    x = re.sub( u"\s+", u" ", x ).strip()

    return x





def keep_only_alpha(w):

    x = re.sub(u"[^a-z]", u" ", w)

    x = re.sub( u"\s+", u" ", x ).strip()

    return x



def preprocess( text ):

    text = re.sub( u"\s+", u" ", text ).strip()

    

    text = re.sub( u"\[math\].*\[\/math\]", u" math ", text) 

    text = re.sub( u"\S*@\S*\.\S*", u" email ", text) 

    

    #replace any integer or real number by the word "number"

    text = u" ".join( re.sub(u"^\d+(?:[.,]\d*)?$", u"number", w)  for w in text.split(" "))

    

    

    specials = [u"’", u"‘", u"´", u"`", u"\u2019"]

    for s in specials:

        text = text.replace(s, u"'")# normalize " ' ", also will be helpful for contractions

        

    text = u" ".join( [contraction_mapping[w] if w in contraction_mapping else w for w in text.split(" ") ] ) 

    

    text = u" ".join( [w if w in embeddings_index else remove_special_chars(w).strip() for w in text.split(" ")] ) 

    

    text = u" ".join( [w if w in embeddings_index else lower(w).strip() for w in text.split(" ")] )

    

    text = u" ".join( [w if w in embeddings_index else keep_alpha_num(w).strip() for w in text.split(" ")] )

    

    text = u" ".join( [w if w in embeddings_index else keep_only_alpha(w).strip() for w in text.split(" ")] )

    

    text = text.split(' ')[:MAX_TEXT_LENGTH]

 

    return ' '.join(text)



def embeddingNN(data,trainable=True, seed=42):                                             

    np.random.seed(seed)



    emb_inpt  = Input( shape=[data.shape[1]], name='emb_inpt')   

    

    #dme

    if len(embedding_weights.shape)==3:

        x1 = Embedding(len( encoding_dc )+1, embedding_weights.shape[1], weights=[embedding_weights[:,:,0]], trainable=trainable) (emb_inpt)

        x2 = Embedding(len( encoding_dc )+1, embedding_weights.shape[1], weights=[embedding_weights[:,:,1]], trainable=trainable) (emb_inpt)

        print (x1.shape, x2.shape)

        x = Lambda( lambda x: K.stack( [x[0],x[1]], axis=-1 ) )([x1,x2]) 

        print (x.shape)

        x = CDME_Block(x, MAX_TEXT_LENGTH, n_emb=embedding_weights.shape[-1])

    else:

        x = Embedding(len( encoding_dc )+1, embedding_weights.shape[1], weights=[embedding_weights], trainable=trainable) (emb_inpt)

        

    x = CuDNNLSTM(64, return_sequences=True) (x)   

    x = GlobalMaxPooling1D()(x)  





    x= Dense(128, trainable=not trainable)(x)

    x = Activation('relu')(x)

    

    x= Dense(1, trainable=not trainable)(x)

    x = Activation('sigmoid')(x)

    

    model = Model([emb_inpt],x)



    return model





def run_model(lr=1e-3, bs=2048):    

    predictions_test   = pd.DataFrame()

    predictions_train  = pd.DataFrame()

    for seed in range(3):

        es = callbacks.EarlyStopping( patience=2 )

        mc = callbacks.ModelCheckpoint( filepath=WEIGHTS_PATH, monitor='val_loss', mode='min', save_best_only=True )



        model = embeddingNN(X_test_emb, trainable=False, seed=seed)

        

        optimizer = optimizers.Adam(lr=lr)

        model.compile(loss="binary_crossentropy", optimizer=optimizer)



        model.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es, mc],

                     batch_size=bs, epochs=1000, verbose=2 )

        ###############################

        es = callbacks.EarlyStopping( patience=2 )

        model = embeddingNN(X_test_emb, trainable=True, seed=seed)

        model.load_weights(WEIGHTS_PATH)

        optimizer = optimizers.Adam(lr=lr/10.)

        model.compile(loss="binary_crossentropy", optimizer=optimizer)



        model.fit(  X_train_emb, y_train, validation_data=(X_test_emb, y_test), callbacks=[es, mc],

                     batch_size=2048, epochs=1000, verbose=2 )

        #######################################

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
#our word index will contain now the union of the vocabulary from glove and paragram



def load_glove_words():

    

    def get_coefs(word,*arr): return word, 1

    

    EMBEDDING_FILE = EMBEDDINGS_PATH+'glove.840B.300d/glove.840B.300d.txt'        

    embeddings_dict = dict()        

    embeddings_dict.update( dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE)) )

    

    EMBEDDING_FILE = EMBEDDINGS_PATH+'paragram_300_sl999/paragram_300_sl999.txt'

    embeddings_dict.update(  dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100) )

    



    return embeddings_dict



embeddings_index = load_glove_words()
X_train, X_test, y_train, y_test = train_test_split(  full_data.question_text.values, full_data.target.values, 

                                                    shuffle =True, test_size=0.5, random_state=42)



X_train = np.array( [preprocess(x) for x in X_train] )

X_test  = np.array( [preprocess(x) for x in X_test] )



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

    



check_coverage(word_frequency_dc,embeddings_index)

print ('number of unique words in the dataset after preprocessing : ', len(word_frequency_dc))


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
EMBEDDING_SIZE = 300



def get_embeddings( word_index , method):

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    

    if method == 'glove':

        EMBEDDING_FILE = EMBEDDINGS_PATH+'glove.840B.300d/glove.840B.300d.txt'

        embeddings = { o.split(" ")[0]:np.asarray(o.split(" ")[1:], dtype='float32') for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index }

    

    if method == 'paragram':

        EMBEDDING_FILE = EMBEDDINGS_PATH+'paragram_300_sl999/paragram_300_sl999.txt'  

        embeddings = { o.split(" ")[0]:np.asarray(o.split(" ")[1:], dtype='float32')\

                      for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore')\

                      if ( (len(o)>100) & (o.split(" ")[0] in word_index) ) }

    

        

    temp = np.stack(embeddings.values())

    mean, std = temp.mean(), temp.std()

  

    embedding_weights    = np.random.normal(mean, std, (len(word_index)+1,  EMBEDDING_SIZE ) ).astype(np.float32)



    for word, i in word_index.items():

        if (word in embeddings):

            embedding_weights[i] = embeddings.get(word)



    return embedding_weights, embeddings

            

            

def load_embeddings(word_index, method='glove'):

    # method is either : 'glove', 'paragram', 'concat', 'avg', 'dme'

    

    if method in [ 'glove' , 'paragram']:

        return get_embeddings( word_index, method )[0]

    else:

        embedding_glove, glove_index       = get_embeddings( word_index, method='glove' )

        embedding_paragram , paragram_index   = get_embeddings( word_index, method='paragram' ) 

        

        if method == 'concat':

            return np.hstack( [embedding_glove, embedding_paragram] )

        if method == 'avg':

            return (embedding_glove + embedding_paragram) / 2.0           

        if method == 'dme':

            return np.stack( [embedding_glove, embedding_paragram], axis=-1 )

        

from keras.layers import Activation

from keras.layers import multiply, Lambda, Reshape

import keras.backend as K

#https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/71778

def CDME_Block(inp, maxlen, n_emb):

    """

    # inp = tensor of shape (?,maxlen,embedding dim,n_emb)) n_emb is number of embedding matrices

    # out = tensor of shape (?,maxlen,embedding dim)

    """

    init = inp

    x = Reshape((maxlen,-1))(inp)

    

    x = CuDNNLSTM(n_emb,return_sequences = True)(x)

    x = Activation('sigmoid')(x)

    x = Reshape((maxlen,1,n_emb))(x)

    x = multiply([init, x])

    out = Lambda(lambda x: K.sum(x, axis=-1))(x)

    return out





embedding_weights = load_embeddings(encoding_dc, method='glove')

print (embedding_weights.shape)

run_model(lr=5e-3, bs=2048)
embedding_weights = load_embeddings(encoding_dc, method='paragram')

print (embedding_weights.shape)

run_model(lr=5e-3, bs=2048)
embedding_weights = load_embeddings(encoding_dc, method='concat')

print (embedding_weights.shape)

run_model(lr=5e-3, bs=2048)
embedding_weights = load_embeddings(encoding_dc, method='avg')

print (embedding_weights.shape)

run_model(lr=5e-3, bs=2048)
embedding_weights = load_embeddings(encoding_dc, method='dme')

print (embedding_weights.shape)

run_model(lr=5e-3, bs=2048)