# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Deep learning
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM, Bidirectional, GlobalMaxPool1D, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.model_selection import train_test_split
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE='../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'
embed_size = 50 # how big is each word vector
max_features = 25000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use
train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
print("Shapes of the datasets ", train.shape, test.shape)
print(train.info())
print(train.sample()['comment_text'])
train['comment_text'] = train['comment_text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
tokenizer = Tokenizer(num_words=max_features, lower=True,split=' ')
tokenizer.fit_on_texts(train['comment_text'].values)
X = tokenizer.texts_to_sequences(train['comment_text'].values)
X_result = tokenizer.texts_to_sequences(test['comment_text'].values)
X = pad_sequences(X, maxlen=maxlen)
X_result = pad_sequences(X_result, maxlen=maxlen)
print(X[0])
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 50))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
Y = pd.get_dummies(train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]).values
print(Y[:10])
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size=32, epochs=2, validation_split=0.1)
y_test = model.predict([X_result], batch_size=1024, verbose=1)
y_test[:10]
sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
sample_submission[['toxic','severe_toxic','obscene','threat','insult','identity_hate']] = y_test
sample_submission.to_csv('submission.csv', index=False)