# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pickle

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.models import Model

CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'

GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words
os.listdir('../input/nlp-getting-started')
sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub.head()
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
test.head()
TEXT_COLUMN = 'text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
x_train = train[TEXT_COLUMN].astype(str)

y_train = train[TARGET_COLUMN].values

x_test = test[TEXT_COLUMN].astype(str)
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)

tokenizer.fit_on_texts(list(x_train) + list(x_test))
crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

print('n unknown words (crawl): ', len(unknown_words_crawl))



glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))
embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

embedding_matrix.shape
MAX_LEN = 300

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
def build_model(embedding_matrix):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=result)

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
NUM_MODELS = 2

BATCH_SIZE = 512

EPOCHS = 5
checkpoint_predictions = []

weights = []



for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix)

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,y_train,

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=2

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=BATCH_SIZE))

        weights.append(2 ** global_epoch)
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
predictions.shape
sub.iloc[:, 1] = (predictions > 0.5).astype(int)
sub.head()
sub.to_csv('submission.csv', index=False)
from collections import Counter

Counter(sub['target'])