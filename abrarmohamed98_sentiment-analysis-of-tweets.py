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
from sklearn.model_selection import train_test_split
tweet=pd.read_csv("../input/Tweets.csv")
X=tweet['text']

Y=tweet['airline_sentiment']
tweet.head()
import re

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

corpus=[]

ps=SnowballStemmer('english')

for i in range(tweet.shape[0]):

  review=re.sub('[^a-zA-Z]',' ',tweet['text'][i])

  review=review.lower()

  review=review.split()

  review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  review=' '.join(review)

  corpus.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(max_features=650)

x=v.fit_transform(corpus).toarray()
x.shape
Y = pd.get_dummies(tweet['airline_sentiment'])
Y
X_train,X_test,Y_train,Y_test = train_test_split(x,Y,test_size=0.2,random_state=10)
from keras.models import Sequential

from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(output_dim=400,init='uniform',activation='relu',input_shape=(650,)))
classifier.add(Dense(output_dim=300,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=200,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=100,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=20,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=3,init='uniform',activation='softmax'))
classifier.summary()
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=100,nb_epoch=100 )
pred=classifier.predict(X_test)
pred
pred=np.argmax(pred, axis=1)
pred=pd.get_dummies(pred)
pred
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, pred)