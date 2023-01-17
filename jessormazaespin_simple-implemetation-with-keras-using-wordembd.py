# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.preprocessing import LabelEncoder

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import os, re, csv, math, codecs

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Masking

from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from tqdm import tqdm

PATH = '../input/google-quest-challenge/'

df_train = pd.read_csv(PATH+'train.csv')

df_test = pd.read_csv(PATH+'test.csv')

sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv').fillna(' ')

sample_submission.head()

class_names = list(sample_submission.columns[1:])

class_question = class_names[:21]

class_answer = class_names[21:]

y1 = df_train[class_question]

y2 = df_train[class_answer]

df_train
le = LabelEncoder()

categoria = df_train.category

train_categoria = le.fit_transform(categoria)

train_categoria = tf.keras.utils.to_categorical(train_categoria, num_classes=5)

x = df_train.columns[[1,2,5]]

x = df_train[x]

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")

tokenizer.fit_on_texts(x.question_title)

tokenizer.fit_on_texts(x.question_body)

tokenizer.fit_on_texts(x.answer)

word_index = tokenizer.word_index

train_title = tokenizer.texts_to_sequences(x.question_title)

train_body = tokenizer.texts_to_sequences(x.question_body)

train_answer = tokenizer.texts_to_sequences(x.answer)

train_title = pad_sequences(train_title)

train_body = pad_sequences(train_body)

train_answer = pad_sequences(train_answer)



print('loading word embeddings...')

embeddings_index = {}

f = codecs.open('../input/fasttext/wiki.simple.vec', encoding='utf-8')

for line in tqdm(f):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('found %s word vectors' % len(embeddings_index))

emb_matrix = np.zeros((1000 + 1, 300))

for word, i in tokenizer.word_index.items():

    if i>= tokenizer.num_words - 1:

        break

    embedding_vector = embeddings_index.get(word)

    if (embedding_vector is not None) and len(embedding_vector) > 0:

        # words not found in embedding index will be all-zeros.

        emb_matrix[i] = embedding_vector

        

def build_model(embedding_matrix):

    embedding = Embedding(

        *embedding_matrix.shape, 

        weights=[embedding_matrix], 

        trainable=False, 

        mask_zero=True

    )

    

    q_in = Input(shape=(None,))

    q = embedding(q_in)

    q = tf.keras.layers.Conv1D(64 // 2,5, padding='valid')(q)

    q = SpatialDropout1D(0.2)(q)

    q = GlobalMaxPooling1D()(q)

    q = tf.keras.layers.Flatten()(q)

    

    t_in = Input(shape=(None,))

    t = embedding(t_in)

    t = tf.keras.layers.Conv1D(64 // 2,5, padding='valid')(t)

    t = SpatialDropout1D(0.2)(t)

    t = GlobalMaxPooling1D()(t)

    t = tf.keras.layers.Flatten()(t)

    a_in = Input(shape=(None,))

    a = embedding(a_in)

    a = tf.keras.layers.Conv1D(64 // 2,5, padding='valid')(a)

    a = SpatialDropout1D(0.2)(a)

    a = GlobalMaxPooling1D()(a)

    a = tf.keras.layers.Flatten()(a)

    c_in = Input(shape=(train_categoria.shape[1]))

    c = Dense(50, activation='relu')(c_in)

    hidden = concatenate([q, t, c])    

    hidden = Dense(300, activation='sigmoid')(hidden)

    out1 = Dense(21, activation='sigmoid')(hidden)

    hidden2 = concatenate([q , a, c])

    hidden2 = Dense(300, activation='sigmoid')(hidden2)

    out2 = Dense(9, activation='sigmoid')(hidden2)

    model = Model(inputs=[t_in, q_in, a_in, c_in], outputs=[out1,out2])

    

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    return model



model = build_model(emb_matrix)

model.summary()
train_target1 = np.array(y1)

train_target2 = np.array(y2)

history = model.fit(

    [train_title, train_body, train_answer, train_categoria],

    [train_target1,train_target2],

    epochs=10,

    validation_split=0.2,

    batch_size=64

)

with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save('modelo.h5')
