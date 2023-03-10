import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



SEED = 2018



np.random.seed(SEED)

tf.random.set_seed(SEED)



from tqdm import tqdm

tqdm.pandas()

import os

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



print(os.listdir("../input/quora-insincere-questions-classification"))
#Caracteres especiales

puncts  = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '???',  '~', '@', '??', 

 '??', '_', '{', '}', '??', '^', '??', '`',  '<', '???', '??', '???', '???', '???',  '???', '???', '??', '??', '???', '???', '??', '???', '??', '??', '???', 

 '???', '???', '???', '???', '???', '??', '???', '???', '??', '??', '??', '???', '??', '???', '??', '??', '???', '???', '??', '???', '???', '??', '???', '???', '???', '???', 

 '???', '???', '??', '???', '???', '???', '???', '???', '???', '???', '??', '???', '???', '???', '??', '??', '???', '??', '???', '??', '??', '??', '??', '???', '???', '???', 

 '???', '???', '???', '???', '???', '???', '??', '???', '???', '???', '???', '??', '???', '???', '???', '???', '???', '???', '???', '??', '??', '??', '???', '???', '???', ]



#Funci??n para quitar del texto los caracteres especiales

def clean_text(x):

    x = str(x)

    for punct in puncts:

        if punct in x:

            x = x.replace(punct, f' {punct} ')

    return x



#Funci??n para limpiar los n??meros

def clean_numbers(x): 

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



#Contracciones comunes

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



#Funci??n para Reemplazar faltas de ortograf??a comunes

def replace_typical_misspell(text): 

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
#Extrayendo data de entrenamiento y validaci??n

train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
#Tama??o training y test set

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
train_df['target'].value_counts().plot(kind = 'pie', labels = ['No Ofensiva', 'Ofensiva'],

     startangle = 100, autopct = '%1.1f%%')
#Quitar caracteres especiales

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))

    

#Limpieza numero

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))

    

#Quitar contracciones

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))



#Pasar letras a minuscula

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())

    

train_X = train_df["question_text"].fillna("_##_").values

splits = list(StratifiedKFold(n_splits=10,random_state=2018).split(train_X,train_df['target'].values))
#Se divide la data en un set de train y uno de validaci??n

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)
#Los valores faltantes se llenan con 'na'

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values
# Configuraci??n

embed_size = 300      # Tama??o m??ximo de cada vector embedding

max_features = 50000  # N??mero total de palabras ??nicas 

maxlen = 100          # N??mero m??ximo de palabras que tiene la pregunta
# Tokenizaci??n de las oraciones

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)
# Si el n??mero de palabras en el texto es mayor que 'max_len' se trunca a 'max_len'. 

# Si el n??mero de palabras en el texto es menor que 'max_len' se agrega ceros para completar valores restantes.

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)
# Consigue los valores objetivos

train_y = train_df['target'].values

val_y = val_df['target'].values
#Se utiliz?? un modelo GRU bidireccional.

inp = tf.keras.layers.Input(shape=(maxlen,))

x = tf.keras.layers.Embedding(max_features, embed_size)(inp)

x = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x)

x = tf.keras.layers.GlobalMaxPooling1D()(x)

x = tf.keras.layers.Dense(16, activation="relu")(x)

x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
bathsize = 512

epoch = 1
model.fit(train_X, train_y, batch_size=bathsize, epochs=epoch, validation_data=(val_X, val_y))
#Ahora se obtiene las predicciones de la muestra de validaci??n y tambi??n se obtiene el mejor umbral para la puntuaci??n de F

pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=0) #verbose=1

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score en el l??mite {0} es {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
#Obtenemos las predicciones y se guardan (no se uso un embedding pre entrenado)

#pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=0)
#Borrar de la memoria 

del model, inp, x

import gc; gc.collect()

time.sleep(10)
!ls ../input/quora-insincere-questions-classification/embeddings/
EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'

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
inp = tf.keras.layers.Input(shape=(maxlen,))

x = tf.keras.layers.Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x)

x = tf.keras.layers.GlobalMaxPooling1D()(x)

x = tf.keras.layers.Dense(16, activation="relu")(x)

x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(train_X, train_y, batch_size=bathsize, epochs=epoch, validation_data=(val_X, val_y))
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=0)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score en el l??mite {0} es {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
#Los resultados parecen ser mejores que el modelo sin incrustaciones pre-entrenadas, guardados las predicciones

#pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=0)
#Borrar de la memoria

del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

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
inp = tf.keras.layers.Input(shape=(maxlen,))

x = tf.keras.layers.Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x)

x = tf.keras.layers.GlobalMaxPooling1D()(x)

x = tf.keras.layers.Dense(16, activation="relu")(x)

x = tf.keras.layers.Dropout(0.1)(x)

x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(train_X, train_y, batch_size=bathsize, epochs=epoch, validation_data=(val_X, val_y))
pred_fasttext_val_y = model.predict([val_X], batch_size=1024, verbose=0)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score en el l??mite {0} es {1}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))))
#pred_fasttext_test_y = model.predict([test_X], batch_size=1024, verbose=1)