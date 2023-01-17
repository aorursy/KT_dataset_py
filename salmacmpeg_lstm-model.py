import csv

import sys

import cv2

import os

import random

import re

import pickle

import numpy as np

import pandas as pd

import tensorflow as tf

from numpy import array

from numpy import asarray

from numpy import zeros

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

from collections import defaultdict

from keras import backend as keras

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D,Embedding,LSTM

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.layers import GlobalMaxPooling1D

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from tensorflow.keras.layers import Embedding



tf.test.gpu_device_name()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

ddir=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        ddir.append(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ddir
def preprocess_train(train):

  #preprocessing for the train dataset

  train['text'] = train['text'].fillna('')

  train['selected_text'] = train['selected_text'].fillna('')

  train['sentiment']=train['sentiment'].replace('neutral',0)

  train['sentiment']=train['sentiment'].replace('positive',1)

  train['sentiment']=train['sentiment'].replace('negative',1)

  #sns.countplot(x='sentiment', data=train)

  #plt.show()

  return train

def preprocess_test(test):

  #preprocessing for the train dataset

  test['text'] = test['text'].fillna('')

  test['sentiment']=test['sentiment'].replace('neutral',0)

  test['sentiment']=test['sentiment'].replace('positive',1)

  test['sentiment']=test['sentiment'].replace('negative',1)

  #sns.countplot(x='sentiment', data=test)

  #plt.show()

  return test
def embed(tokenizer):

  embeddings_dictionary = dict()

  glove_file = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.50d.txt', encoding="utf8")



  for line in glove_file:

      records = line.split()

      word = records[0]

      vector_dimensions = asarray(records[1:], dtype='float32')

      embeddings_dictionary [word] = vector_dimensions

  glove_file.close()



  #embeddings_dictionary

  embedding_matrix = zeros((vocab_size, 50))

  for word, index in tokenizer.word_index.items():

      embedding_vector = embeddings_dictionary.get(word)

      if embedding_vector is not None:

          embedding_matrix[index] = embedding_vector

  return embedding_matrix

#analysing

train_original=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train_original=preprocess_train(train_original)



test_original=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_original=preprocess_test(test_original)
tokenizer = Tokenizer(num_words=500000,filters='')

tokenizer.fit_on_texts(train_original['text'])

maxlen=35

train_original['Text_Sequences'] = pd.Series(tokenizer.texts_to_sequences(train_original['text']))

train_original['Text_Sequences_padded']  = pad_sequences(train_original['Text_Sequences'] , padding='post', maxlen=maxlen).tolist()



test_original['Text_Sequences'] = pd.Series(tokenizer.texts_to_sequences(test_original['text']))

test_original['Text_Sequences_padded']  = pad_sequences(test_original['Text_Sequences'] , padding='post', maxlen=maxlen).tolist()



train_original['Selected_Text_Sequences']=pd.Series(tokenizer.texts_to_sequences(train_original['selected_text']))

train_original['Selected_Text_Sequences_padded'] = pad_sequences(train_original['Selected_Text_Sequences'], padding='post', maxlen=maxlen).tolist()



Y_train_per_word=np.zeros((train_original['Text_Sequences_padded'].shape[0],maxlen))

idx=0

for sentence in train_original['Text_Sequences_padded']:

  idx2=0

  for word in sentence:

    #print(word)

    if (word != 0) and (train_original['sentiment'][idx]!=0):

      if (word==train_original['Text_Sequences_padded'][idx][idx2]):

        Y_train_per_word[idx][idx2]= train_original['sentiment'][idx]

        idx2=idx2+1

  idx=idx+1

train_original['Y_labeled']=Y_train_per_word.tolist()

#train_original['jaccard_distance']=jaccard_score(train_original['Text_Sequences_padded'],train_original['Selected_Text_Sequences_padded'])
X_train_temp=(train_original[train_original['sentiment']!=0]['Text_Sequences_padded'])

X_train=np.array([np.array(xi) for xi in X_train_temp])

Y_train_temp=train_original[train_original['sentiment']!=0]['Y_labeled']

Y_train=np.array([np.array(xi) for xi in Y_train_temp])
vocab_size = len(tokenizer.word_index) + 1

embedding_matrix=embed(tokenizer)
def jaccard_distance(y_true, y_pred, smooth=100):

    """ Calculates mean of Jaccard distance as a loss function """

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1))

    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1))

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    jd =  (1 - jac) * smooth

    return tf.reduce_mean(jd)
def jaccard_score(y_true, y_pred, smooth=100):

    """ Calculates mean of Jaccard distance as a loss function """

    arr1=np.array([np.array(xi) for xi in y_true])

    arr1=(arr1>0).astype('int')

    arr2=np.array([np.array(xi) for xi in y_pred])

    arr2=(arr2>0).astype('int')

    intersection = np.sum(np.multiply(arr1 , arr2), axis=(1))

    sum_ =np.sum(arr1,axis=1)+np.sum(arr2,axis=1)

    jac = (intersection + smooth) / (sum_ - intersection + smooth) #score

    jd =  (1 - jac) * smooth #distance

    return (jd)


X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)

model1 = Sequential()

embedding_layer = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=maxlen , trainable=False)

model1.add(embedding_layer)

model1.add(Dropout(0.2))

model1.add((Dense(35, activation='relu')))

model1.add(LSTM(128,input_shape=(maxlen,50), return_sequences=True))

model1.add(Dropout(0.2))

model1.add((Dense(1, activation='sigmoid')))

model1.compile(optimizer='adam', loss=jaccard_distance)

print(model1.summary())



y_train1 = y_train1.reshape(y_train1.shape[0], y_train1.shape[1], 1)

y_test1 = y_test1.reshape(y_test1.shape[0], y_test1.shape[1], 1)

#history = model1.fit(X_train1, y_train1, batch_size=128, epochs=75, verbose=1, validation_data=(X_test1, y_test1))

history = model1.fit(X_train, Y_train, batch_size=128, epochs=75, verbose=1, validation_data=(X_test1, y_test1))

score = model1.evaluate(X_test1, y_test1, verbose=1)

print(score)
def postprocessing1(res2):

  resres2=np.zeros((res2.shape))

  i=0

  for item in res2:

    arr=np.where(item==True)[0]

    first=arr[0]

    length=len(np.where(item==True)[0])

    last=arr[length-1]

    for j in range(first,last+1):

      resres2[i][j]=1

    i=i+1

  return resres2



def postprocessing2(X_TEST,TEST_SENTIMENT,RES_Y_TEST):

  X_submission=np.copy(X_TEST)

  i=0

  for item in RES_Y_TEST:

    if TEST_SENTIMENT[i]!=0:

      j=0

      for word in item:

        if word==0 and X_TEST[i][j]!=0:

          X_submission[i][j]=0 

        j=j+1

    i=i+1

  return X_submission
y_pred=model1.predict(X_test1)

res=y_pred>0.5

res=postprocessing1(res)

jac_des=jaccard_score(res,y_test1)

print("mean",np.mean(jac_des))

X_test_temp=(test_original['Text_Sequences_padded'])

X_TEST=np.array([np.array(xi) for xi in X_test_temp])

Y_test_temp=test_original['sentiment']

TEST_SENTIMENT=np.array([np.array(xi) for xi in Y_test_temp])

Y_TEST=model1.predict(X_TEST)

RES_Y_TEST=Y_TEST>0.5

RES_Y_TEST=postprocessing1(RES_Y_TEST)

X_submission=postprocessing2(X_TEST,TEST_SENTIMENT,RES_Y_TEST)

X_submission_text=tokenizer.sequences_to_texts(X_submission)

#X_submission_text
from pandas import DataFrame

df = DataFrame({'textID': test_original['textID'], 'selected_text': X_submission_text})

df.to_csv('submission.csv', index=False)