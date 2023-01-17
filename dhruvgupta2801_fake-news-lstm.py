# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/fake-news/train.csv')

train_data.head()
train_data['label'].value_counts()
train_data.shape
train_data=train_data.dropna()

print(train_data.shape)
X=train_data.drop('label',axis=1)

X.head()
y=train_data['label']
test_data=pd.read_csv('/kaggle/input/fake-news/test.csv')

test_data.head()
from tensorflow.keras.layers import Embedding,Flatten,Dense,LSTM

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot
vocab_size=5000
import regex as re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
messages=X.copy()

messages.reset_index(inplace=True)

messages.head()
ps=PorterStemmer()

corpus_train=[]

for i in range(len(messages)):

    review=re.sub('[^a-zA-Z]',' ',messages['title'][i])

    review=review.lower()

    review=review.split()

    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]

    review=' '.join(review)

    corpus_train.append(review)
corpus_train[0:10]
messages['title'][0:10]
one_hot_rep=[one_hot(words,vocab_size) for words in corpus_train]

one_hot_rep[0]
sent_length=20

embedded_docs=pad_sequences(one_hot_rep,padding='pre',maxlen=sent_length)

embedded_docs[0]
embedded_features=40

model=Sequential()

model.add(Embedding(vocab_size,embedded_features,input_length=sent_length))

model.add(LSTM(100))

model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
X_final=np.asarray(embedded_docs)

y_final=np.asarray(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,random_state=42)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
train_model=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10)
import matplotlib.pyplot as plt
plt.plot(train_model.history['accuracy'],'b',label='train_accuracy')

plt.plot(train_model.history['val_accuracy'],'r',label='val_accuracy')

plt.legend()
def test(X):

    X=X.fillna(0)

    messages=X.copy()

    messages.reset_index(inplace=True)

    corpus=[]

    for i in range(len(messages)):

        reviews=re.sub('[^a-zA-Z]',' ',str(messages['title'][i]))

        reviews=reviews.lower()

        reviews=reviews.split()

        reviews=[ps.stem(word) for word in reviews if word not in stopwords.words('english')]

        reviews=' '.join(reviews)

        corpus.append(reviews)

    one_hot_rep=[one_hot(word,vocab_size)for word in corpus]

    embedded_docs = pad_sequences(one_hot_rep, padding = 'pre', maxlen = sent_length)

    X_final = np.array(embedded_docs)

    

    

    return X_final
test_data_new= test(test_data)

test_data_new[0]
pred=model.predict(test_data_new)

print(pred.shape)
submission_data = pd.read_csv('/kaggle/input/fake-news/submit.csv')
submission_data['label']=np.round(pred).astype('int')
submission_data.head()
submission_data.to_csv('Submission.csv',index=False)