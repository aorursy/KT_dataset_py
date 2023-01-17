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
import keras

import tensorflow as tf

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import pandas as pd





df = pd.read_csv('../input/training.1600000.processed.noemoticon.csv',engine='python',header=None)
df.columns=['target','id','date','flag','user','text']

df = df.sample(frac=1).reset_index(drop=True)
df_n = pd.get_dummies(df['target'])

df=df.drop(axis=1,labels='target')

df['positive']=df_n[4]



import re

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

corpus=[]

ps=SnowballStemmer('english')

for i in range(df.shape[0]):

  review=re.sub('[^a-zA-Z]',' ',df['text'][i])

  review=review.lower()

  review=review.split()

  review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  review=' '.join(review)

  corpus.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(max_features=650)

x=v.fit_transform(corpus).toarray()
x_train=x[:1300000,:]

y_train=df.iloc[:1300000,-1].values

x_test=x[130000:,:]

y_test=df.iloc[130000:,-1].values

from keras.models import Sequential

from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(output_dim=300,init='uniform',activation='relu',input_shape=(650,)))
classifier.add(Dense(output_dim=200,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=100,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=20,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=32500,nb_epoch=35)
y_pred=classifier.predict(x_test)
pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test, pred))

print('\n')

print(classification_report(y_test, pred))