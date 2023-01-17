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

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

import itertools

from sklearn.metrics import accuracy_score, confusion_matrix
#read the data

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

df_submission=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

test.head()
#divide the data

train=train.dropna()

test=test.dropna()

train_X=train['selected_text']

train_y=train['sentiment']

test_X=test['text']

test_y=test['sentiment']

null_values = train.isnull().sum(axis=0)

print(null_values)
#now apply tfidfvectorizer

tfidf = TfidfVectorizer(stop_words='english',max_df=0.4)



#fit and transform training set and transform test set

tfidf_train=tfidf.fit_transform(train_X)

tfidf_test=tfidf.transform(test_X)

#print(tfidf_train)
#now initialize the passiveAggressiveClassifier

pac = PassiveAggressiveClassifier(max_iter=135)

pac.fit(tfidf_train, train_y)

pred=pac.predict(tfidf_test)

print(pred)
#check accuracy

score=accuracy_score(test_y,pred)

print(score)
df_submission['selected_text'] = test['text']

df_submission.to_csv("submission.csv", index=False)

display(df_submission.head(10))