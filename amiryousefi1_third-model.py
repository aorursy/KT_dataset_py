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
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
### Text Normalizing function
def clean_text(text): 
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text
print("Clean Text Function defined")
print("Start Cleaning Data texts")
train_df["question_text"] = train_df["question_text"].map(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].map(lambda x: clean_text(x))
print("Data Questions Text Cleaned")
## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
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
print("Creating Model by GLOVE Embeddings")
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
print("Starting Prediction by GLOVE Embeddings for VAL")
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)
best_thresh = 0.5
best_score = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
    if score > best_score:
        best_thresh = thresh
        best_score = score
print()
print("Best Tresh for GLOVE is {0} at score {1}".format(best_thresh,best_score))
print()
pred_glove_val_y = (pred_glove_val_y>best_thresh)
print("Prediction for test based on GLOVE Embeddings")
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_glove_test_y = (pred_glove_test_y>best_thresh)
print("Glove Emebddings Prediction Complete")
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)
print("Creating Model by Wikinews Embeddings")
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
print("Starting Prediction by wikinews Embeddings for VAL")
pred_fasttext_val_y = model.predict([val_X], batch_size=1024, verbose=1)
best_thresh = 0.5
best_score = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))))
    if score > best_score:
        best_thresh = thresh
        best_score = score
print()
print("Best Tresh for wikinews is {0} at score {1}".format(best_thresh,best_score))
print()
pred_fasttext_val_y = (pred_fasttext_val_y>best_thresh)
print("Prediction for test based on wikinews Embeddings")
pred_fasttext_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_fasttext_test_y = (pred_fasttext_test_y>best_thresh)
print("Wikinews Emebddings Prediction Complete")
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)
print("Creating Model by paragram embeddings")
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
print("Starting Prediction by paragram Embeddings")
pred_paragram_val_y = model.predict([val_X], batch_size=1024, verbose=1)
best_thresh = 0.5
best_score = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int))))
    if score > best_score:
        best_thresh = thresh
        best_score = score
print()
print("Best Tresh for paragram is {0} at score {1}".format(best_thresh,best_score))
print()
pred_paragram_val_y = (pred_paragram_val_y>best_thresh)
print("Prediction for test based on paragram Embeddings")
pred_paragram_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_paragram_test_y = (pred_paragram_test_y>best_thresh)
print("Paragram Emebddings Prediction Complete")
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)
pred_val_y = 0.34*pred_glove_val_y + 0.33*pred_fasttext_val_y + 0.33*pred_paragram_val_y 

best_thresh = 0.5
best_score = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
    if score > best_score:
        best_thresh = thresh
        best_score = score
print()
print("Best Tresh for overall is {0} at score {1}".format(best_thresh,best_score))
print()
pred_val_y = (pred_val_y>best_thresh)
pred_test_y = 0.34*pred_glove_test_y + 0.33*pred_fasttext_test_y + 0.33*pred_paragram_test_y
pred_test_y = (pred_test_y>best_thresh).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
print("Submission Created")