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
df=pd.read_csv('../input/mbti_1.csv')

df.head()
import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

p=PorterStemmer()

stwrds=stopwords.words("english")

def filtr(st):

    arr=[re.sub("http.*","",x) for x in st.split() if x not in stwrds]

    arr=[p.stem(x) for x in arr]

    return ' '.join(arr)

    
types=df['type'].unique()

df['category']=df['type'].apply(lambda x: np.where(types==x)[0][0])

df['posts'].apply(filtr)

df.groupby('category').count()

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(df['posts'],df['category'],shuffle=True,test_size=0.3)

print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
ytrain.unique().shape #check if training set contains all categories
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=10000,ngram_range=(1,2))

cv.fit(xtrain)

xtrain1=cv.transform(xtrain)

#try using logistic regression first

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix

lr=LogisticRegression()

lr.fit(xtrain1,ytrain)

pred=lr.predict(cv.transform(xtest))

print(accuracy_score(ytest,pred))

print(confusion_matrix(ytest,pred))
ytrain1=pd.get_dummies(ytrain)
from keras.models import Sequential

from keras.layers import Dense,Dropout

model=Sequential()

model.add(Dense(500,input_shape=(10000,),activation='relu'))

model.add(Dropout(0.6))

model.add(Dense(16,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(xtrain1,ytrain1,epochs=20)
model.evaluate(cv.transform(xtest),pd.get_dummies(ytest))
import pickle

pickle.dump(cv.vocabulary_,open("vocab.pkl","wb"))

model.save('mymodel.h5')

!ls
from keras.models import load_model

tcv=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("vocab.pkl","rb")))

m=load_model('mymodel.h5')
!mv 'mymodel.h5' '../input/mymodel.h5'
m.evaluate(tcv.transform(xtest),pd.get_dummies(ytest))
#API keys from developer.twitter.com

apikey='7NCzIYyE6v4rXHFgBfMjy6GqD'

apisecretkey='mkD5V5WdViePkoiNdt5R4W0o8PJ8tojHXUxGgMzHhctp0rbRI6'

acctoken='1144102706333618176-wIUa115mxUw7n1IHmpFEW5RTSs81li'

acctokensecret='ccwvQxyexrKQTpdzRJyxgF6nJratx2qvKzTbWrrMwgPYn'
import tweepy

api=tweepy.API(apikey,apisecretkey)