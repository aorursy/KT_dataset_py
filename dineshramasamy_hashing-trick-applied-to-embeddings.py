import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras import regularizers
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big should each word vector be
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

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
train_y = train_df['target'].values
val_y = val_df['target'].values
!ls ../input/embeddings/
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
compressive_mat = np.random.randn(embed_size // 3 , all_embs.shape[1])

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size // 3))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix1[i] = compressive_mat.dot(embedding_vector)
        
del embeddings_index
import gc; gc.collect()
time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, 
                                                               encoding="utf8", errors='ignore'))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
compressive_mat = np.random.randn(embed_size // 3, all_embs.shape[1])

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size // 3 ))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix2[i] = compressive_mat.dot(embedding_vector)
        
del embeddings_index
import gc; gc.collect()
time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
compressive_mat = np.random.randn(embed_size // 3, all_embs.shape[1])

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size // 3 ))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix3[i] = compressive_mat.dot(embedding_vector)
        
del embeddings_index
import gc; gc.collect()
time.sleep(10)
embedding_matrix = np.concatenate([embedding_matrix1, embedding_matrix2, embedding_matrix3],axis=1)

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)

f_scores = []
for thresh in np.arange(0.1, 0.9, 0.01):
    thresh = np.round(thresh, 2)
    f_score = metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))
    f_scores.append((f_score, thresh))
    
fmax, opt_thresh = max(f_scores)
print("F1 score at threshold {0} is {1}".format(opt_thresh, fmax))

pred_all_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = pred_all_test_y
pred_test_y = (pred_test_y>opt_thresh).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)