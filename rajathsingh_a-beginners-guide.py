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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.shape
test_df.shape
# Finding NaN

print(train_df.isna().sum()) 

print(test_df.isna().sum())
# Finding the no of 0 (sincere) and 1 (insincere)

train_df.target.value_counts()
from sklearn.model_selection import train_test_split

train, val = train_test_split(train_df, test_size = 0.2, random_state = 654)
# Separating Questions

X_train = train['question_text']

X_valid = val['question_text']

X_test = test_df['question_text']
from keras.preprocessing.text import Tokenizer

max_features = 50000    # We give a num of unique words to the tokenizer.

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)

X_valid = tokenizer.texts_to_sequences(X_valid)

X_test =  tokenizer.texts_to_sequences(X_test)
# for occurence of words

tokenizer.word_counts
# for index of words

tokenizer.word_index
maxlen = 100 # The max len for a question to be



from keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, maxlen = maxlen)

X_valid = pad_sequences(X_valid, maxlen = maxlen)

X_test =  pad_sequences(X_test, maxlen = maxlen)
Y_train = train['target']

Y_valid = val['target']
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, layers, regularizers, constraints, optimizers
# We need to provide the exact size of the questions as shape to the Input that we decided before using padding

# By leaving out space after comma, we tell keras to conclude a number itself.

inp = Input(shape=(maxlen, ))



# 300 is embed size, how big is each word vector. To understand, NLP book Recipe 3-7.

x = Embedding(max_features, 300)(inp) 



# We place this tensor into LSTM Layer. To understand LSTM https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714

x = LSTM(64, return_sequences=True, name = 'lstm_layer')(x)



# Now we need to reduce the dimension from 3D to 2D and we do it using the GlobalMaxPool.

x = GlobalMaxPool1D()(x)



# We add a Demse layer to our network. 16 is the output produced by the Dense Layer.

x = Dense(16, activation='relu')(x) 



# Dropout layer. We use it so that our NN doesn't overfit.

x = Dropout(0.1)(x)



# This is the final dense layer that gives the output. We need one output for our question 0 or 1.

# The reason why we use sigmoid is bec this is a binary classification problem.

x = Dense(1, activation='sigmoid')(x)
# Providing the inputs and outputs for our model

model = Model(inputs = inp, outputs = x)

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.summary()
# Fitting the model

model.fit(X_train, Y_train, batch_size=512, epochs=2, validation_data=(X_valid, Y_valid))
pred = model.predict([X_test], batch_size=1024, verbose=1)
pred_test_y = (pred>0.35).astype(int)

output_df = pd.DataFrame({"qid":test_df["qid"].values})

output_df['prediction'] = pred_test_y

output_df.to_csv("submission.csv", index=False)