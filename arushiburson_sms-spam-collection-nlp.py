import nltk

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords
messages = [line for line in open('../input/SMSSpamCollection')]

messages[50]
len(messages)
messages = pd.read_csv('../input/SMSSpamCollection', sep='\t', names=['label', 'message'])

messages.head()
messages.describe()
messages.groupby('label').describe()
messages['length'] = messages['message'].apply(len)

messages.head()
sns.set_style('darkgrid')

messages['length'].plot.hist(bins=50)
messages['length'].describe()
messages.hist(column='length', by='label', figsize = (10,4), bins=40)
import string

def clean_data(mess):

    """

    This function removes punctuation and stopwords 

    and returns a list of clean words.

    """

    #removing punctuation

    mess = [item for item in mess if item not in string.punctuation]

    mess = "".join(mess)

    #removing stopwords

    clean = [word for word in mess.split() if word.lower() not in stopwords.words('english')]

    return clean
from sklearn.model_selection import train_test_split

#splitting dataset

messages_train, messages_test, labels_train, labels_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

messages_train.shape, messages_test.shape
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

pipeline = Pipeline([('bow', CountVectorizer(analyzer=clean_data)), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB()) ])

pipeline.fit(messages_train, labels_train)
prediction = pipeline.predict(messages_test)

prediction[0]
from sklearn.metrics import classification_report

print(classification_report(labels_test, prediction))
#using random forests

from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([('bow', CountVectorizer(analyzer=clean_data)), ('tfidf', TfidfTransformer()), ('rfc', RandomForestClassifier())])

pipeline.fit(messages_train, labels_train)
prediction = pipeline.predict(messages_test)

prediction[0]
print(classification_report(labels_test, prediction))
#using logistic regression

from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([('bow', CountVectorizer(analyzer=clean_data)), ('tfidf', TfidfTransformer()), ('classifier', LogisticRegression())])

pipeline.fit(messages_train, labels_train)
prediction = pipeline.predict(messages_test)

print(classification_report(labels_test, prediction))