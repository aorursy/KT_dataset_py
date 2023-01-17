import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math



from sklearn.metrics import roc_curve, auc,  f1_score

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import keras

from sklearn import metrics

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, BatchNormalization, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,CuDNNLSTM

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Bidirectional, GlobalMaxPool1D, TimeDistributed

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.preprocessing import text, sequence



from nltk.corpus import stopwords

import string 



from keras.callbacks import ModelCheckpoint

import re
!du -l ../input/*
df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

#df['non_toxic'] = df.apply(lambda x: 1 if x.toxic == 0 & x.severe_toxic == 0 & x.obscene == 0 & x.threat == 0 & x.insult == 0 & x.identity_hate == 0 else 0 , axis=1)
test_df =  pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

#test_y =   pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

#print(test_y)
stop_words = set(stopwords.words('english'))

df['comment_text'] = df.apply(lambda x: x.comment_text.lower(), axis=1)

df['comment_text'] = df.apply(lambda x: re.sub(r'\d+', '', x.comment_text), axis=1)

df['comment_text'] = df.apply(lambda x: x.comment_text.translate(str.maketrans('', '', string.punctuation)),axis=1)

df['comment_text'] = df.apply(lambda x: x.comment_text.strip(),axis=1)

df['comment_text'] = df.apply(lambda x: x.comment_text.rstrip(),axis=1)
test_df['comment_text'] = test_df.apply(lambda x: x.comment_text.lower(), axis=1)

test_df['comment_text'] = test_df.apply(lambda x: re.sub(r'\d+', '', x.comment_text), axis=1)

test_df['comment_text'] = test_df.apply(lambda x: x.comment_text.translate(str.maketrans('', '', string.punctuation)),axis=1)

test_df['comment_text'] = test_df.apply(lambda x: x.comment_text.strip(),axis=1)

test_df['comment_text'] = test_df.apply(lambda x: x.comment_text.rstrip(),axis=1)
#filename = '../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin.gz'

#model = KeyedVectors.load_word2vec_format(filename, binary=True)





train_df, val_df = train_test_split(df, test_size=0.2, random_state=2018)

#test_df =  pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')



## some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## fill up the missing values

train_X = train_df["comment_text"].fillna("_na_").values

val_X = val_df["comment_text"].fillna("_na_").values

test_X = test_df["comment_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)





## Get the target values

train_y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat',

       'insult', 'identity_hate']].values

val_y = val_df[['toxic', 'severe_toxic', 'obscene', 'threat',

       'insult', 'identity_hate']].values

#test_y = test_df[['toxic', 'severe_toxic', 'obscene', 'threat',

#       'insult', 'identity_hate','non_toxic']].values


EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'



max_features= 50000

maxlen = 100

embed_size = 300





def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')

    

def get_embed_mat(EMBEDDING_FILE, max_features,embed_dim):

    # word vectors

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))

    print('Found %s word vectors.' % len(embeddings_index))



    # embedding matrix

    word_index = tokenizer.word_index

    num_words = min(max_features, len(word_index) + 1)

    all_embs = np.stack(embeddings_index.values()) #for random init

    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 

                                        (num_words, embed_dim))

    for word, i in word_index.items():

        if i >= max_features:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    max_features = embedding_matrix.shape[0]

    

    return embedding_matrix



embedding_matrix = get_embed_mat(EMBEDDING_FILE,max_features,300)

print(embedding_matrix.shape)
filter_sizes = [1,2,3,5,10]

num_filters = 128



inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Reshape((maxlen, embed_size, 1))(x)



maxpool_pool = []

for i in range(len(filter_sizes)):

    conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),

                                 kernel_initializer='glorot_uniform', activation='relu')(x) 

    maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))



z = Concatenate(axis=1)(maxpool_pool) 

z = TimeDistributed(Bidirectional(CuDNNLSTM(256)))(z)

z = SpatialDropout1D(.5)(z)

z = Bidirectional(CuDNNLSTM(256))(z)

z = Dropout(.5)(z)

z = BatchNormalization()(z)

#z = Flatten()(z)

z = Dense(1000, activation="relu")(z)

z = Dropout(.5)(z)

z = Dense(500, activation="relu")(z)

z = Dropout(.5)(z)

z = Dense(100, activation="relu")(z)



outp = Dense(6, activation="softmax")(z)



model = Model(inputs=inp, outputs=outp)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()



#from keras.callbacks import EarlyStopping, ModelCheckpoint

#earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

mcp_save = ModelCheckpoint('../input/mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_X, train_y, batch_size=512, epochs=30, validation_data=(val_X, val_y), callbacks = [mcp_save])
y_pred = model.predict(test_X)

print(y_pred)
df_final = pd.DataFrame(np.concatenate((np.array(test_df.id).reshape(-1,1),y_pred[:,:6]),axis=1), columns = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
df_final.head()
df_final.to_csv('toxic_v2.csv',index=False)