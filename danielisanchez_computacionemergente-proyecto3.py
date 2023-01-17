# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import tensorflow as tf



SEED = 2018



np.random.seed(SEED)

tf.set_random_seed(SEED)



from tqdm import tqdm

tqdm.pandas()



import os

print(os.listdir("../input"))



#Librerias

import os

import time

import numpy as np 

import pandas as pd 

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, BatchNormalization

from keras.optimizers import Adam, Nadam

from keras.models import Model

from keras import backend as K

from keras.callbacks import Callback

from keras import initializers, regularizers, constraints, optimizers, layers

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.engine.topology import Layer

import gc, re

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, auc, precision_recall_curve
import time

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, BatchNormalization

from keras.optimizers import Adam, Nadam

from keras.models import Model

from keras import backend as K

from keras.callbacks import Callback



from keras import initializers, regularizers, constraints, optimizers, layers



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



from keras.engine.topology import Layer





import gc, re

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, auc, precision_recall_curve
#Limpiamos el texto, tanto los caracteres especiales como las contracciones y se reemplaza en el texto



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):

    x = str(x)

    for punct in puncts:

        if punct in x:

            x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",              

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"ain't" :  "will not",

"didn't": "did not"}



mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
#Limpiar texto

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))

    

#Limpio numero

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))

    

#Limpio contracciones

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

    

train_X = train_df["question_text"].fillna("_##_").values

splits = list(StratifiedKFold(n_splits=10,random_state=2018).split(train_X,train_df['target'].values))
## Se divide para entrenar

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



# Algunas config de los valores

embed_size = 300 # tamano max de cad avector

max_features = 50000 # numero de palabras unicas

maxlen = 100 # num max de palabras que tiene la pregunta



# Se complentan los valores faltantes

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values



# Tokenizan las oraciones

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



# Rellenan las oraciones

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



# Consigue los valores objetivos

train_y = train_df['target'].values

val_y = val_df['target'].values
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
# Entrenando el modelo

model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del model, inp, x

import gc; gc.collect()

time.sleep(10)
!ls ../input/embeddings/
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
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
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
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_fasttext_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))))
pred_fasttext_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
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
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_paragram_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int))))
pred_paragram_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
pred_val_y = 0.33*pred_glove_val_y + 0.33*pred_fasttext_val_y + 0.34*pred_paragram_val_y 

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
pred_test_y = 0.33*pred_glove_test_y + 0.33*pred_fasttext_test_y + 0.34*pred_paragram_test_y

pred_test_y = (pred_test_y>0.35).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)
# pred_test_y = np.sum([outputs[i][1] * reg.coef_[i] for i in range(len(outputs))], axis = 0)

coefs = [0.35,0.25,0.2,0.2]

# pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis = 0)

pred_test_y = np.sum([outputs[i][1]*coefs[i] for i in range(len(coefs))], axis = 0)



pred_test_y = (pred_test_y > results['threshold']).astype(int)

test_df = pd.read_csv("../input/test.csv", usecols=["qid"])

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)