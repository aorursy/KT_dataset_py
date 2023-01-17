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

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,CuDNNLSTM

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

import re
import matplotlib.pyplot as plt

import seaborn as sns

import gc



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

from sklearn.model_selection import train_test_split

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from scipy import sparse

%matplotlib inline

seed = 42
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)


def clean_text(x):

    #translator = str.maketrans('', '', string.punctuation)

    x = x.lower()

    #return x.translate(translator)

    x = re.sub(r"[^\sa-zA-Z']", "", x)

    return x

#train_df['question_text']= train_df['question_text'].apply(lambda x: clean_text(x))
#test_df['question_text']=test_df['question_text'].apply(lambda x: clean_text(x))
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 90000 # how many unique words to use (i.e num rows in embedding vector)

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
#from imblearn.over_sampling import SMOTE

#train_X, train_y = SMOTE().fit_resample(train_X, train_y)



#val_X, val_y = SMOTE().fit_resample(val_X, val_y)
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(train_y),train_y)
class_weight_dict = dict(enumerate(class_weights))
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))#, class_weight=class_weight_dict)
# Get training and test loss histories

training_loss = history.history['loss']

test_loss = history.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_loss, 'r--')

plt.plot(epoch_count, test_loss, 'b-')

plt.legend(['Training Loss', 'Test Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show();
#val_X_n = val_df["question_text"].fillna("_na_").values

#val_X_n = tokenizer.texts_to_sequences(val_X_n)

#val_X_n = pad_sequences(val_X_n, maxlen=maxlen)

#val_y_n = val_df['target'].values
pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.701, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
pred_nval_y = (pred_noemb_val_y>0.35).astype(int)

pred_nval_y
from sklearn import metrics
print(metrics.classification_report(pred_nval_y, val_y))
print(metrics.accuracy_score(pred_nval_y, val_y))
print(metrics.f1_score(pred_nval_y, val_y))
print(metrics.recall_score(pred_nval_y, val_y))
confusion_matrix= metrics.confusion_matrix(pred_nval_y, val_y)
confusion_matrix
sns.heatmap(confusion_matrix, annot=True, cmap=sns.color_palette("Paired"))
pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)

pred_test_y = (pred_noemb_test_y>0.35).astype(int)

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

x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history=model.fit(train_X, train_y, batch_size=512, epochs=5, validation_data=(val_X, val_y))
# Get training and test loss histories

training_loss = history.history['loss']

test_loss = history.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_loss, 'r--')

plt.plot(epoch_count, test_loss, 'b-')

plt.legend(['Training Loss', 'Test Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show();
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history=model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
# Get training and test loss histories

training_loss = history.history['loss']

test_loss = history.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_loss, 'r--')

plt.plot(epoch_count, test_loss, 'b-')

plt.legend(['Training Loss', 'Test Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show();
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.01, 0.501, 0.01):

    thresh = np.round(thresh, 3)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
pred_glove_val_y = (pred_glove_val_y>0.37).astype(int)

print(metrics.classification_report(pred_glove_val_y, val_y))
print(metrics.accuracy_score(pred_glove_val_y, val_y))
print(metrics.f1_score(pred_glove_val_y, val_y))
print(metrics.recall_score(pred_glove_val_y, val_y))
confusion_matrix_glove= metrics.confusion_matrix(pred_glove_val_y, val_y)
sns.heatmap(confusion_matrix_glove, annot=True, cmap=sns.color_palette("Paired"))
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del model, inp, x

import gc; gc.collect()

time.sleep(10)
pred_glove_test_y = (pred_glove_test_y>0.33).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)


