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
import re

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, chi2

from sqlite3 import Error

from sklearn.ensemble import RandomForestClassifier

import sqlite3

import pickle

import nltk

nltk.download('stopwords')

%matplotlib inline
dataset = pd.read_csv('../input/bbc-text.csv')

dataset.head()
dataset.groupby('category').text.count().plot.bar(ylim=0)

plt.show()
nltk.download('stopwords')

stemmer = PorterStemmer()

words = stopwords.words("english")

dataset['cleaned'] = dataset['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
dataset.head()
vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))

final_features = vectorizer.fit_transform(dataset['cleaned']).toarray()

final_features.shape
from sklearn.linear_model import LogisticRegression

X = dataset['cleaned']

Y = dataset['category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)



pipeline = Pipeline([('vect', vectorizer),

                     ('chi',  SelectKBest(chi2, k=1200)),

                     ('clf', LogisticRegression(random_state=0))])



model = pipeline.fit(X_train, y_train)

with open('LogisticRegression.pickle', 'wb') as f:

    pickle.dump(model, f)



ytest = np.array(y_test)



# confusion matrix and classification report(precision, recall, F1-score)

print(classification_report(ytest, model.predict(X_test)))

print(confusion_matrix(ytest, model.predict(X_test)))


