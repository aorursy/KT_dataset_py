# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
tqdm.pandas()
%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gensim.models.keyedvectors as word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, Dropout, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import os
print(os.listdir("../input"))
import gensim.models.keyedvectors as word2vec
import gc
# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head(10)
def build_vocab(sentences,verbose=True):
    vocab={}
    
    for sentence in tqdm(sentences,disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

sentences = train["question_text"].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})
from gensim.models import KeyedVectors
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
word2vecDict= KeyedVectors.load_word2vec_format(news_path, binary=True)

import operator 

def check_coverage(vocab,word2vecDict):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = word2vecDict[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
oov = check_coverage(vocab,word2vecDict)
oov[:15]
def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x
                    
train_df["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
test_df["question_text"] = test["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)
oov = check_coverage(vocab,word2vecDict)
oov[:16]
for i in range(10):
    print(word2vecDict.index2entity[i])
import re

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}',"##",x)
    return x
train_df["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
test_df["question_text"] = test["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train_df["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)
oov = check_coverage(vocab,word2vecDict)
oov[:20]
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
train_df["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
test_df["question_text"] = test["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)
oov = check_coverage(vocab,word2vecDict)
oov[:20]

del(oov)

gc.collect()
train.head(20)
embed_size = 300
maxlen = 200
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)

train_x,val_x=train_test_split(train_df, test_size=0.1, random_state=2018)
train_X=train_x["question_text"].fillna("_na_").values
val_X=val_x["question_text"].fillna("_na_").values
test_X=test_df["question_text"].fillna("_na_").values

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
train_y = train_x['target'].values
val_y = val_x['target'].values
#word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin", binary=True)

embeddings_index = dict()
for word in word2vecDict.wv.vocab:
    embeddings_index[word] = word2vecDict.word_vec(word)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_features: continue
    if word in word2vecDict:
        embedding_vector = word2vecDict.get_vector(word)
        embedding_matrix[i] = embedding_vector
        
del word2vecDict; gc.collect()   
#inp = Input(shape=(maxlen,))
#x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
#x = GlobalMaxPool1D()(x)
#x = Dense(16, activation="relu")(x)
#x = Dropout(0.1)(x)
#x = Dense(1, activation="sigmoid")(x)
#model = Model(inputs=inp, outputs=x)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
embed_size = 300 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 200 # max number of words in a question to use

S_DROPOUT = 0.4
DROPOUT = 0.1


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size , weights=[embedding_matrix_3])(inp)
x = SpatialDropout1D(S_DROPOUT)(x)
x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
x = Dense(16, activation="relu")(conc)
x = Dropout(DROPOUT)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))


embed_size = 300 
max_features = 50000 
maxlen = 100 

train_x,val_x=train_test_split(train, test_size=0.1, random_state=42)
train_X=train_x["question_text"].fillna("_na_").values
val_X=val_x["question_text"].fillna("_na_").values
test_X=test["question_text"].fillna("_na_").values
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
train_y = train_x['target'].values
val_y = val_x['target'].values


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
model.fit(train_X, train_y, batch_size=1024, epochs=2, validation_data=(val_X, val_y))
pred_test_y = model.predict([val_X], batch_size=512, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_test_y>thresh).astype(int))))
pred_y = model.predict([test_X], batch_size=512, verbose=1)

pred_y = (pred_y>0.35).astype(int)
out= pd.DataFrame({"qid":test["qid"].values})
out['prediction'] = pred_y
out.to_csv("submission.csv", index=False)


