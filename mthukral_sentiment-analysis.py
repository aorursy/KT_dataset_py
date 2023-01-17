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




dataset = pd.read_csv("../input/IMDB Dataset.csv")

dataset.head()

X = dataset.iloc[:,0:1].values

y = dataset.iloc[:,1:2].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
import re

import nltk

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

corpus_test = []

for i in range(len(X_test)):

    cleanr = re.compile('<.*?>')

    review = re.sub(cleanr, '', X_test[i][0])

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review  if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus_test.append(review)
corpus_train = []

for i in range(len(X_train)):

    cleanr = re.compile('<.*?>')

    review = re.sub(cleanr, '', X_train[i][0])

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review  if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus_train.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,1))

X_train_after = tfidf.fit_transform(corpus_train).toarray()
X_test_after = tfidf.transform(corpus_test).toarray()
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()

y_train = labelencoder_y.fit_transform(y_train.ravel())
y_test_test = labelencoder_y.fit_transform(y_test.ravel())
y_train
X_train_after
# 1st model Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train_after, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test_after)

y_pred
y_test
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_test, y_pred)
cm
from sklearn.metrics import accuracy_score

accuracy_score(y_test_test, y_pred)