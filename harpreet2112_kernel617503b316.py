# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import gc

import csv

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk import TweetTokenizer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, GRU, LSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

use_gpu=True
train = pd.read_csv("/kaggle/input/question2/train.csv",sep=';')

test = pd.read_csv("/kaggle/input/question2/test.csv",sep=';',quoting=csv.QUOTE_NONE)

del test['Unnamed: 1']
train['tweet'] = train['tweet'].apply(lambda x : ' '.join([w for w in x.split() if not w.startswith('@') ])  ) 

test['tweet'] = test['tweet'].apply(lambda x : ' '.join([w for w in x.split() if not w.startswith('@') ])  )
full_text = list(train['tweet'].values) + list(test['tweet'].values)

full_text = [i.lower() for i in full_text if i not in stopwords.words('english') and i not in ['.',',','/','@','"','&amp','<br />','+/-','zzzzzzzzzzzzzzzzz',':-D',':D',':P',':)','!',';']]
y = train['sentiment']
tk = Tokenizer(lower = True, filters='')

tk.fit_on_texts(full_text)



train_tokenized = tk.texts_to_sequences(train['tweet'])

test_tokenized = tk.texts_to_sequences(test['tweet'])



max_len = 50

X_train = pad_sequences(train_tokenized, maxlen = max_len)

X_test = pad_sequences(test_tokenized, maxlen = max_len)
embedding_path ="/kaggle/input/embedding/glove.6B.100d.txt"



embed_size =100 

max_features = 30000



def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path , encoding="utf8"))
word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re

# Create check-point

file_path = "best_model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

embed_dim = 128

lstm_out = 196



model = Sequential()

model.add(Embedding(max_features, embed_size,input_length = X_train.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())

history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 15, validation_split=0.1, 

                        verbose = 1, callbacks = [check_point, early_stop])
pred = model.predict(X_test, batch_size = 1024)
twt = ["I had asked him to put together a performance report for Emami as Suraj Saharan was meeting the client. The report Ranjith put together was only pivot tables (which was also wrong) and there was absolutely no analysis. Similar case in another client where we were facing an issue and Ranjith seemed more interested in arguing with the team instead of trying to understand the problem and resolve it. Requires a lot of improvement"]

#vectorizing the tweet by the pre-fitted tokenizer instance

twt = tk.texts_to_sequences(twt)

#padding the tweet to have exactly the same shape as `embedding_2` input

twt = pad_sequences(twt, maxlen=50, dtype='int32', value=0)

#print(twt)

sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]

if(np.argmax(sentiment)==0):

    print("negative")

elif (np.argmax(sentiment)==1):

    print("positive")