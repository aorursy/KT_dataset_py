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
import pandas as pd

import matplotlib.pyplot as plt

import re

import nltk

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score


test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')



#data shape

print("Test data has {} rows and {} columns ".format(test_df.shape[0], test_df.shape[1]))

print("Train data has {} rows and {} columns ".format(train_df.shape[0], train_df.shape[1]))
#check for null values

print("Null for test data :", test_df.isnull().sum())

print("Null for train data :", train_df.isnull().sum())

#Target variable

print("Distribution of target variable :", train_df['target'].value_counts())



sns.barplot(x = train_df['target'].value_counts().index, y = train_df['target'].value_counts() )

plt.show()
count_vectorizer = CountVectorizer()


train_vector = count_vectorizer.fit_transform(train_df['text']).todense()

test_vector = count_vectorizer.transform(test_df['text']).todense()


print("train vector :", train_vector.shape)

print("test vector :", test_vector.shape)
X = train_vector

y = train['target'].values



print("X shape :", X.shape)

print("y shape :", y.shape)



print("test vector shape :", test_vector.shape)



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =2020)



logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)



y_pred_lm = logistic_model.predict(X_test)



f1score = f1_score(y_test, y_pred_lm)

print("Logistic Model f1: {}".format(f1score*100))



accuracy_score = accuracy_score(y_test, y_pred_lm)

print("Logistic model accuracy score is {}".format(accuracy_score))
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sample_submission['target'] = logistic_model.predict(test_vector)

print("Submission head: ", sample_submission.head(10))

sample_submission.to_csv('submission.csv', index=False)