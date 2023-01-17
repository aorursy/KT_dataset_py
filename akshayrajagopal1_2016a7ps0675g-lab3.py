import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.layers import Dropout

import re

import sys, os, re, csv, codecs, numpy as np, pandas as pd



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

#from nltk.corpus import stopwords

#from nltk import word_tokenize

#STOPWORDS = set(stopwords.words('english'))

from bs4 import BeautifulSoup

#import plotly.graph_objs as go

#import plotly.plotly as py

import cufflinks

from IPython.core.interactiveshell import InteractiveShell

#import plotly.figure_factory as ff

InteractiveShell.ast_node_interactivity = 'all'

#from plotly.offline import iplot

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
train = pd.read_csv('../input/nnfl-lab-3-nlp/nlp_train.csv')

test = pd.read_csv('../input/nnfl-lab-3-nlp/_nlp_test.csv')

em_file=f'../input/glove6b50dtxt/glove.6B.50d.txt'

pd.set_option('display.max_colwidth',-1)

train.head()
# size of word vector

em_size = 50



# embedding vector feature count

max_features = 23914 



# max number of words from a single comment

maxlen = 300 

list_train = train["tweet"].fillna("_na_").values

list_classes = [0, 1, 2, 3]

y = train['offensive_language']

list_test = test["tweet"].fillna("_na_").values
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_test)

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
def get_coefficients(word,*arr): 

    return word, np.asarray(arr, dtype='float32')



em_ind = dict(get_coefficients(*t.strip().split()) for t in open(em_file))
em_full = np.stack(em_ind.values())

em_mean = em_full.mean() 

em_std = em_full.std()

print(em_mean,em_std)
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

em_mat = np.random.normal(em_mean, em_std, (nb_words, em_size))

for word, i in word_index.items():

    if i >= max_features: 

        continue

    em_vec = em_ind.get(word)

    if em_vec is not None: 

        em_mat[i] = em_vec
inp = Input(shape=(maxlen,))

x = Embedding(max_features, em_size, weights=[em_mat])(inp)

x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1)(x)

model = Model(inputs=inp, outputs=x)
X_train, X_test, y_train, y_test = train_test_split(X_t,y, test_size = 0.1, random_state = 42)

X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size = 0.5, random_state = 42)



print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
model.compile(loss = 'mse', optimizer='adam',metrics = ['mae', 'mse'])
model.fit(X_t, y, batch_size = 64, epochs = 6)
pred = model.predict(X_val)
predictions = np.round_(pred)

from sklearn.metrics import mean_squared_error as mse

error = mse(y_val, pred)

error
from math import sqrt
err = sqrt(error)

err
pred = model.predict(X_test)
predictions = np.round_(pred)
error = mse(y_test, pred)

error
pred = model.predict(X_te)

submission = test
submission.head()
submission["offensive_language"] = pred
submission.head()
model.save_weights("2016a7ps0675g_model_lab3.h5")
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(submission)