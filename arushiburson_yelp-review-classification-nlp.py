#importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#importing the dataset

yelp = pd.read_csv('../input/yelp.csv')

yelp.head()
yelp.info()
yelp.describe()
#creating new column to use length of text as a feature

yelp["text length"] = yelp["text"].apply(len)

yelp.head()
g = sns.FacetGrid(yelp,col='stars')

g.map(plt.hist,'text length')
sns.boxplot(y='text length', data=yelp, x='stars')
sns.countplot(yelp['stars'])
stars = yelp.groupby('stars')['cool', 'useful', 'funny', 'text length'].mean()

stars
stars_corr = stars.corr()

stars_corr
sns.heatmap(stars_corr, cmap='coolwarm', annot=True)
#creating a dataframe yelp_class such that it contains only 1 star and 5 star reviews

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

yelp_class.head()
X = yelp_class['text']

y = yelp_class['stars']
#using bag of words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_cv = cv.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)
prediction_nb = nb.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, prediction_nb))

print(confusion_matrix(y_test, prediction_nb))
#using custom analyzer with bag of words model

import string

from nltk.corpus import stopwords

def clean_data(mess):

    """

    This function removes punctuation and stopwords 

    and returns a list of clean words.

    """

    mess = [item for item in mess if item not in string.punctuation]

    mess = "".join(mess)

    clean = [word for word in mess.split() if word.lower() not in stopwords.words('english')]

    return clean

cv2 = CountVectorizer(analyzer = clean_data)

X_cv2 = cv2.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_cv2, y, test_size=0.3, random_state=101)

nb.fit(X_train, y_train)

prediction_nb2 = nb.predict(X_test)

print(classification_report(y_test, prediction_nb2))

print(confusion_matrix(y_test, prediction_nb2))
#using TFIDF in order to verify if there is a change in the algorithm performance

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

#creating a pipeline

pipeline = Pipeline([('cv', CountVectorizer()), ('tfidf', TfidfTransformer()), ('nb', MultinomialNB())])
#splitting test and train data again

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#fitting data to te pipeline

pipeline.fit(X_train, y_train)
#making prediction

prediction_pl = pipeline.predict(X_test)
#evaluating performance

print(classification_report(y_test, prediction_pl))

print(confusion_matrix(y_test, prediction_pl))
#using logistic regression on bag of words

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()

logr.fit(X_train, y_train)

predict_logr = logr.predict(X_test)

print(classification_report(y_test, predict_logr))

print(confusion_matrix(y_test, predict_logr))
#using random forests on bag of words

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.3, random_state=101)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

predict_rfc = rfc.predict(X_test)

print(classification_report(y_test, predict_rfc))

print(confusion_matrix(y_test, predict_rfc))