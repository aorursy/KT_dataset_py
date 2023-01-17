# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import sys, os, re, csv, codecs, numpy as np, pandas as pd



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
path = '../input/'

comp = 'jigsaw-toxic-comment-classification-challenge/'

EMBEDDING_FILE='../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'

TRAIN_DATA_FILE=f'{path}{comp}train.csv'

TEST_DATA_FILE=f'{path}{comp}test.csv'
#set parameters

embed_size = 100 # how big is each word vector

max_features = 25000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a comment to use
#read datasets

train = pd.read_csv(TRAIN_DATA_FILE)

test = pd.read_csv(TEST_DATA_FILE)
list_sentences_train = train["comment_text"].fillna("_na_").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_test = test["comment_text"].fillna("_na_").values
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

emb_mean,emb_std
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)

x = Dense(100, activation="relu")(x)

x = Dropout(0.25)(x)

x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_t, y, batch_size=32, epochs=10) # validation_split=0.1);
y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')

sample_submission[list_classes] = y_test

sample_submission.to_csv('submission.csv', index=False)
#took help of this https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout-lb-0-048  