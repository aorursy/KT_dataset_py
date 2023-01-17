import os

import re

import nltk

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer





train = pd.read_csv('../input/train.csv')





test = pd.read_csv('../input/test.csv')
inp=train.comment_text

Y=train.target
def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data



x_train = preprocess(inp)

x_test = preprocess(test.comment_text)

from sklearn.feature_extraction.text import TfidfVectorizer



# create the transform

vectorizer = TfidfVectorizer()

# tokenize and build vocab

vectorizer.fit_transform(x_train.astype('U'))

#vectorizer.fit(x)

# summarize

##print(vectorizer.vocabulary_)

##print(vectorizer.idf_)

# encode document

x_train = vectorizer.fit_transform(x_train.astype('U'))

# summarize encoded vector

print(x_train.shape)
from sklearn.feature_extraction.text import TfidfVectorizer



# create the transform

##vectorizer = TfidfVectorizer()

# tokenize and build vocab

##vectorizer.fit_transform(x_test.astype('U'))

#vectorizer.fit(x)

# summarize

##print(vectorizer.vocabulary_)

##print(vectorizer.idf_)

# encode document

x_test = vectorizer.transform(x_test.astype('U'))

# summarize encoded vector

print(x_test.shape)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
y_train = np.where(train['target'] >= 0.5, 1, 0)
y_train
classifier.fit(x_train, y_train)
predictions = classifier.predict_proba(x_test)[:, 1]
predictions
submission = pd.read_csv('../input/sample_submission.csv')



submission['prediction'] = predictions



submission.to_csv('submission.csv',index=False)