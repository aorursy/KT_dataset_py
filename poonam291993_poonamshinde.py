import sklearn

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

import seaborn as sas

import pandas as pd

Tweets = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")

Tweets.head()
len(Tweets)-Tweets.count()/len(Tweets)
dt_count=Tweets['airline_sentiment'].value_counts()

dt_count
index=[1,2,3]

plt.bar(index,dt_count)

plt.xticks(index,['negative','positive','neutral'])

plt.xlabel('dt')

plt.ylabel('dt count')

plt.title('count of dt')

plt.plot()
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix 
labels=Tweets['airline_sentiment']

labels.head()
X_train,X_test,y_train,y_test=train_test_split(Tweets['text'],labels,test_size=0.6,random_state=7)

tfidf_vect=TfidfVectorizer(stop_words='english',max_df=0.6)

t_train=tfidf_vect.fit_transform(X_train)

t_test=tfidf_vect.transform(X_test)

pact=PassiveAggressiveClassifier(max_iter=100)

pact.fit(t_train,y_train)

y_pred=pact.predict(t_test)

score=accuracy_score(y_test,y_pred)

print(f'Accuracy:{round(score*100,2)}%')

confusion_matrix(y_test,y_pred,labels=['negative','positive','neutral'])
Tweets['tweet_id']