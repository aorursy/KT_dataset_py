# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
toxic_dir = "../input/jigsaw-toxic-comment-classification-challenge"

quora_dir = "../input/quora-insincere-questions-classification"



print(os.listdir(toxic_dir))
import pandas as pd

import numpy as np

import gc
train = pd.read_csv(toxic_dir + "/train.csv")

test_df = pd.read_csv(toxic_dir + "/test.csv")



train.head()
print(train.shape)

train.info()
from sklearn.model_selection import train_test_split



## split to train and val

train_df, val_df = train_test_split(train, test_size=0.1, random_state=2019)

train_df.head()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train_df[list_classes].values

y_val = val_df[list_classes].values

print(y_train.shape,y_val.shape)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



## fill up the missing values

train_X = train_df["comment_text"].fillna("_na_").values

val_X = val_df["comment_text"].fillna("_na_").values

test_X = test_df["comment_text"].fillna("_na_").values



maxlen = 50

max_features = 50000



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
def get_coefs(word,*arr): 

     return word, np.asarray(arr, dtype='float32')
embeddings_dir = quora_dir + "/embeddings"

os.listdir(embeddings_dir)
EMBEDDING_FILE_1 = embeddings_dir + '/glove.840B.300d/glove.840B.300d.txt'

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_1))



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))



# Creating the embedding matrix with GLoVE

embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix_1[i] = embedding_vector



del embeddings_index; gc.collect()
print(embedding_matrix_1.shape)
EMBEDDING_FILE_2 = embeddings_dir + '/wiki-news-300d-1M/wiki-news-300d-1M.vec'

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_2) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))



# Creating the embedding matrix with wikinews embeddings

embedding_matrix_2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix_2[i] = embedding_vector

        

del embeddings_index; gc.collect() 
print(embedding_matrix_2.shape)
embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2), axis=1)  



del embedding_matrix_1, embedding_matrix_2

gc.collect()



print("Shape of embedding matrix: ",np.shape(embedding_matrix))
# Importing necessary libraries

from keras.layers import Dense, Input, CuDNNLSTM, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers
spatial_dropout = 0.4



inp = Input(shape=(maxlen,))



x = Embedding(max_features, embed_size * 2, weights=[embedding_matrix])(inp)

x = SpatialDropout1D(spatial_dropout)(x)

#x = Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True))(x)

x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)



avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

conc = concatenate([avg_pool, max_pool])



x = Dense(16, activation="relu")(conc)

x = Dropout(0.2)(x)

x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)



model.summary()
model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
batch_size = 512

epochs = 5



hist = model.fit(train_X, y_train, batch_size=batch_size, epochs=epochs, validation_data=(val_X, y_val))
y_test = model.predict([test_X], batch_size=1024, verbose=1)

sample_submission = pd.read_csv(toxic_dir + "/sample_submission.csv")

sample_submission[list_classes] = y_test

sample_submission.to_csv('submission.csv', index=False)