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
#importing common libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
# creating a dataframe of the data



test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test_labels = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')
train.head()
test.head()
test_labels.head()
train.isnull().any()
test.isnull().any()
train.columns.values
list_classes = ['toxic', 'severe_toxic', 'obscene', 'threat',

       'insult', 'identity_hate']

y = train[list_classes].values



list_sentences_train = train['comment_text']

list_sentences_test = test['comment_text']
from keras.preprocessing.text import Tokenizer



max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))



list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
list_tokenized_train[:1]

from keras.preprocessing.sequence import pad_sequences

max_len = 200



X_train = pad_sequences(list_tokenized_train,maxlen=max_len)

X_test = pad_sequences(list_tokenized_test,maxlen=max_len)
NumWords = [len(com) for com in list_tokenized_train]
sns.distplot(NumWords,bins=50)
from keras.layers import Dense,Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
inp = Input(shape= (max_len,))
from keras.models import Sequential



embed_size = 128



model = Sequential()

model.add(Embedding(max_features,embed_size))

model.add(LSTM(60,return_sequences=True, name='lstm_layer'))

model.add(GlobalMaxPool1D())

model.add(Dropout(0.1))

model.add(Dense(50,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(6,activation='sigmoid'))



model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
batch_size = 32

epochs=1



model.fit(X_train,y,batch_size=batch_size,epochs=epochs,validation_split=0.1)

predictions = model.predict(X_test)
predictions
len(predictions)
pred = pd.DataFrame(predictions,columns=list_classes)
pred.head()
sub = pd.concat([test['id'], pred],axis=1)
sub.head()
sub.to_csv('out.csv',index=False)
