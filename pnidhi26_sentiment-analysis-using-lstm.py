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
train = pd.read_csv('../input/dataset/train.tsv', delimiter='\t')

test = pd.read_csv('../input/dataset/test.tsv', delimiter='\t')
train.head()
train.describe()

train.head()
x_train = train['Phrase']

y_train = train['Sentiment']

x_train.head()
sns.countplot(y_train)

y_train.value_counts()
print("Phrase:-", x_train[1],"\nSentiment:-", y_train[1])
tokenizer = Tokenizer(num_words=15000)

tokenizer.fit_on_texts(list(x_train))
X_train = tokenizer.texts_to_sequences(x_train)

X_train = pad_sequences(X_train, maxlen=150)

X_train.shape
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

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.summary()
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=4, batch_size=1000, verbose=1)
test = tokenizer.texts_to_sequences(test['Phrase'])

x_test = pad_sequences(test, maxlen=150)
y = model.predict(x_test)

y = np.argmax(y, axis=1)
sub = pd.read_csv('../input/sampleSubmission.csv')

sub['Sentiment'] = y

sub.to_csv('output.csv', index=False)