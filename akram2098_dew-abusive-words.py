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
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense,Dropout,Embedding,LSTM,Conv2D,Flatten,MaxPooling2D

import matplotlib.pyplot as plt

import seaborn as sns

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.describe()
x_train = train['Comment']

y_train = train['Insult']
sns.countplot(y_train)

y_train.value_counts()
print("Comment:-", x_train[1],"\nInsult:-", y_train[1])
tokenizer = Tokenizer(num_words=15000)

tokenizer.fit_on_texts(list(x_train))
X_train = tokenizer.texts_to_sequences(x_train)

X_train = pad_sequences(X_train, maxlen=150)

X_train
Y = to_categorical(y_train.values)

Y
train_x, val_x, train_y, val_y = train_test_split(X_train, Y, test_size=0.2)
train_x.shape
train_y.shape
val_x.shape
val_y.shape
model=Sequential()

model.add(Embedding(15000,512,mask_zero=True))

model.add(LSTM(512,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))

model.add(LSTM(256,dropout=0.1, recurrent_dropout=0.1,return_sequences=False))

model.add(Dense(5,activation='softmax'))

model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.summary()
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=4, batch_size=200, verbose=1)
y = model.predict(train_x)

y = np.argmax(y, axis=1)

y
y1 = model.predict(val_x)

y1= np.argmax(y1, axis=1)

y1