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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re
data = pd.read_excel("../input/all-data-clean.xlsx")

data = data[['cleanText','label']]

data['cleanText'].head()

txt = data['cleanText']

lab = data['label']
max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(txt.values)

X = tokenizer.texts_to_sequences(txt.values)

X = pad_sequences(X)
embed_dim = 128

lstm_out = 196



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
Y = pd.get_dummies(lab).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
batch_size = 1000

model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)
validation_size = 1500



X_validate = X_test[-validation_size:]

Y_validate = Y_test[-validation_size:]

X_test = X_test[:-validation_size]

Y_test = Y_test[:-validation_size]

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))