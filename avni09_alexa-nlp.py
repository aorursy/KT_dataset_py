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
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
import nltk
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = [line.rstrip() for line in open('../input/amazon_alexa.tsv')]
print (len(df))
df = pd.read_csv('../input/amazon_alexa.tsv', sep='\t')
df.head()
df.describe().T
df.dtypes
df.verified_reviews[10]
X = df.verified_reviews
y = df.rating
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = TfidfVectorizer(stop_words='english')
dtm = vect.fit_transform(df.verified_reviews)
features = vect.get_feature_names()
dtm.shape
review = TextBlob(df.verified_reviews[105])
review
review.words
import nltk
# list the sentences
review.sentences

stemmer = SnowballStemmer('english')
# stem each word
print ([stemmer.stem(word) for word in review.words])
nltk.download('wordnet')
print ([word.lemmatize() for word in review.words])
# Function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    text = text.lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]
split_into_lemmas
vect = CountVectorizer(analyzer=split_into_lemmas)
# Function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print ('Features: ', X_train_dtm.shape[1])
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print ('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
tokenize_test(vect)
print (vect.get_feature_names()[-50:])
print (vect.get_feature_names()[50:])
# polarity ranges from -1 (most negative) to 1 (most positive)
review.sentiment.polarity
# define a function that accepts text and returns the polarity
def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity
# create a new DataFrame column for sentiment (WARNING: SLOW!)
df['sentiment'] = df.verified_reviews.apply(detect_sentiment)
df.boxplot(column='sentiment', by='rating')
df[df.sentiment == 1].verified_reviews.head()
df[df.sentiment == -1].verified_reviews.head()
# negative sentiment in a 5-star review
df[(df.rating == 5) & (df.sentiment < -0.3)].head(1)
# positive sentiment in a 1-star review
df[(df.rating == 1) & (df.sentiment > 0.5)].head(1)
feature_cols = ['verified_reviews', 'variation', 'feedback', 'sentiment']
X = df[feature_cols]
y = df.rating
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train.verified_reviews)
X_test_dtm = vect.transform(X_test.verified_reviews)
print (X_train_dtm.shape)
print (X_test_dtm.shape)
X_train.drop('verified_reviews', axis=1).shape
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
print (metrics.accuracy_score(y_test, y_pred_class))