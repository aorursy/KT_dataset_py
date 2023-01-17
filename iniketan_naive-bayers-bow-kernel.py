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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

subm = pd.read_csv('../input/sample_submission.csv')
train.head()
train['comment_text'][2]
lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()

lens.hist()
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
len(train), len(test)
corpus = []

for i in range(159570):

    review = re.sub('[^a-zA-Z]', ' ', train['comment_text'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = (' ').join(review)

    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train[list_classes]

X_train = cv.fit_transform(train.comment_text).toarray()

X_test = cv.transform(test.comment_text)
from sklearn.naive_bayes import GaussianNB

submisssion = []

for label in list_classes:

    classifire = GaussianNB()

    classifire.fit(X_train, y_train[label])

    submission[label] = classifire.predict_prob(X_test)[:, 1]
submission.to_csv('submission_1.csv', index=False)