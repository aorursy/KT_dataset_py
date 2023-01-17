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

import glob

from sklearn.model_selection import train_test_split

import re

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import accuracy_score
eng_words=pd.read_csv("/kaggle/input/english_text.csv")

eng_words
hin_words=pd.read_csv("/kaggle/input/hinglish_text.csv")

hin_words
#deleting ID column fron dataset

eng_words.drop(['ID'],axis=1,inplace=True)

hin_words.drop(['ID'],axis=1,inplace=True)
eng_words.head(5)
hin_words.head(5)
eng_words['text_language']='english'
eng_words.head(5)
hin_words['text_language']='hinglish'
hin_words.head(5)
#combining both english and hinglish data

words=eng_words.append(hin_words)
words
words['text_language'].value_counts()
words['text']=words['text'].str.lower()
words.head(10)
words.loc[words['text_language']=='english','text_language',]=0

words.loc[words['text_language']=='hinglish','text_language',]=1
words
x=words.text

y=words.text_language
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
import tensorflow as tf

from tensorflow import keras
from __future__ import absolute_import, division, print_function, unicode_literals



!pip install -q tensorflow-hub

!pip install -q tfds-nightly

import tensorflow_hub as hub

import tensorflow_datasets as tfds
#here using pre-trained text embedding model from TensorFlow Hub 



embed_model = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"

first_lay = hub.KerasLayer(embed_model, input_shape=[], 

                           dtype=tf.string, trainable=True)
first_lay(x_train[2:8])
model = tf.keras.Sequential()

model.add(first_lay)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1))



model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
x_train
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


x_tf = tf.convert_to_tensor(x_train)

print(x_tf)
y_tf = tf.convert_to_tensor(y_train)

y_tf
model_train = model.fit(x_tf, y_tf, epochs=20, verbose=1)
y_train=y_train.astype('int')
logreg = LogisticRegression()

logreg.fit(first_lay(x_train), y_train)
y_pred = logreg.predict(first_lay(x_test))

y_pred
x_test[25:35]
y_pred[25:35]
acc_log = round(logreg.score(first_lay(x_train), y_train) * 100, 2)

print (str(acc_log) + ' percent')
logreg.score(first_lay(x_train), y_train)
logreg.score(first_lay(x_test),y_test.astype('int'))
print(classification_report(y_test.astype('int'),y_pred))
accuracy_score(y_test.astype('int'),y_pred)