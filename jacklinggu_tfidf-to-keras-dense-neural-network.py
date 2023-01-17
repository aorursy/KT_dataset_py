# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
DATA_FILE = '../input/spam.csv'

df = pd.read_csv(DATA_FILE,encoding='latin-1')

print(df.head())
tags = df.v1

texts = df.v2
# change y to int

y = [1 if tmp_y=='ham' else 0 for tmp_y in tags]

print(y[:10])
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=100)

tfidf_mat = vec.fit_transform(texts).toarray()

print(type(tfidf_mat),tfidf_mat.shape)  # 5572 doc, tfidf 100 dimension
# build model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D





model = Sequential()

model.add(Dense(64,input_shape=(100,)))

model.add(Dropout(0.2))

model.add(Activation('relu'))

model.add(Dense(64))

model.add(Dropout(0.2))

model.add(Activation('relu'))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['acc'])
model.fit(tfidf_mat,y,batch_size=32,epochs=10,verbose=1,validation_split=0.2)