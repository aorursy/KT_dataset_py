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
dataset = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
dataset.head()
dataset.drop(['tweet_id','airline_sentiment_confidence','negativereason','negativereason_confidence','airline'],axis=1,inplace=True)
dataset.drop(['airline_sentiment_gold','name','negativereason_gold','retweet_count','tweet_coord','tweet_created','tweet_location','user_timezone'],axis=1,inplace=True)
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,14640):

    review = re.sub(r'@\w+', ' ', dataset['text'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)

corpus
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()

y = dataset['airline_sentiment']
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

model.fit(X_train,y_train)



y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



cm = confusion_matrix(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)
acc
cm