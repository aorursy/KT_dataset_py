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
import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train_data=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

test_data=pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
train_data.head()
print(train_data.info())
print(test_data.info())

import sys,os,re,csv,codecs

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense,Conv1D,LSTM,Embedding,Dropout,Bidirectional,MaxPooling1D,GlobalMaxPooling1D

from keras.models import Sequential

r=train_data.copy()

r.drop(columns=['id','comment_text'],inplace=True)

class_names=list(r.columns)

print(class_names)
labels_train=train_data[class_names].values

features_train=train_data['comment_text']

features_test=test_data['comment_text']
max_limit_of_words=30000

tokenizer=Tokenizer(num_words=max_limit_of_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)

tokenizer.fit_on_texts(list(features_train))
token_train=tokenizer.texts_to_sequences(features_train)
token_test = tokenizer.texts_to_sequences(features_test)

word_index = tokenizer.word_index

vocab_size = len(word_index)

mean = np.mean([len(i) for i in token_train])

std = np.std([len(i) for i in token_train])

maximum=int(mean+std*3)



aX_train = pad_sequences(token_train,maxlen=maximum,padding='post',truncating='post')

aX_test=pad_sequences(token_test,maxlen=maximum,padding='post',truncating='post')
dim=100

ei={}

file=open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt',encoding='utf-8')

for line in file:

    vals=line.rstrip().rsplit(' ',dim)

    word = vals[0]

    coefs = np.asarray(vals[1:], dtype='float32')

    ei[word] = coefs

file.close()

print('Found {} word vectors.'.format(len(ei)))
em = np.zeros((vocab_size +1,dim))

tokens = []

labels = []



for word,i in word_index.items():

    temp = ei.get(word)

    if temp is not None:

        em[i] = temp
embedding_layer = Embedding(len(word_index)+1,dim,input_length=maximum,weights=[em])
model=Sequential()

model.add(embedding_layer)

model.add(Bidirectional(LSTM(30,return_sequences=True,dropout = 0.1 , recurrent_dropout = 0.1)))

model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling1D(3))

model.add(GlobalMaxPooling1D())

model.add(Dense(50, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(6, activation='sigmoid'))



model.summary()
model.summary()
from sklearn.model_selection import train_test_split

X_train, X_cross_val, Y_train,Y_cross_val = train_test_split(aX_train, labels_train,test_size=0.30,shuffle=True)
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=800, epochs=1,validation_data=(X_cross_val, Y_cross_val),verbose=1, shuffle=True )
test_labels = model.predict([aX_test], batch_size=800, verbose=1)

mysubmission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

mysubmission[class_names] = test_labels

mysubmission.to_csv('submission.csv')