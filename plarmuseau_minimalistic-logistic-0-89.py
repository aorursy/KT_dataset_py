import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#1: unreliable
#0: reliable
train=pd.read_csv('../input/deceptive-opinion.csv')
train.info()
train
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

#data prep
train['total']=train['hotel']+' '+train['text']
#tfidf
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(train['total'].values)
tfidf = transformer.fit_transform(counts)
targets = train['deceptive'].values

#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


logreg.fit(X_test, y_test)
print('Accuracy of Logistic regression classifier on reversed training set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
print('Accuracy of Logistic regression classifier on reversed test set: {:.2f}'
     .format(logreg.score(X_train, y_train)))


logreg.fit(tfidf, targets)
print('Accuracy of Logistic regression classifier on total training set: {:.2f}'
     .format(logreg.score(tfidf, targets)))
