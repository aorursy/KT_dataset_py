# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
with open('../input/train.json') as datafile:

    data = json.load(datafile)

df = pd.DataFrame(data)

df.head()
X = df['ingredients'].values

y = df['cuisine'].values

print(X)
a = {}

for i in range(len(X)):

    for j in X[i]:

        if j not in a:

            a[j] = [y[i]]

        else:

            a[j].append(y[i])
b = {}

for i in a:

    b[i] = len(a[i])

c = sorted(b.items(), key=lambda x: x[1])

print("Top 5 used ingredients", c[-1:-6:-1])

print("Least used ingredients", c[0:5])
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 39774):

    message = re.sub('[^a-zA-Z_0-9]', ' ', " ".join([item.replace(' ', '') for item in X[i]]))

    message = message.lower()

    message = message.split()

    ps = PorterStemmer()

    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]

    message = ' '.join(message)

    corpus.append(message)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 10000)

X = cv.fit_transform(corpus).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, shuffle=True)

# Fitting classifier to the Training set

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

pre = mnb.predict(X_test)

print(accuracy_score(pre, y_test))
