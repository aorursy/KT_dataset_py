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
import numpy as np

import pandas as pd

import itertools

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data

df=pd.read_csv("../input/airline-tweets-sentiments.csv")

#Get shape and head

labels=df.tweet_sentiment_value

labels.head()



x_train,x_test,y_train,y_test=train_test_split(df['tweet_text'], labels, test_size=0.6, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.6)

# Fit and transform train set, transform test set

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 

tfidf_test=tfidf_vectorizer.transform(x_test)



# Initialize a PassiveAggressiveClassifier

pac=PassiveAggressiveClassifier(max_iter=100)

pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy

y_pred=pac.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test,y_pred, labels=['Negative','Positive'])
df
df['tweet_sentiment_value'].value_counts()