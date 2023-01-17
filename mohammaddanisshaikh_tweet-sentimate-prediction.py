

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
#Load data

data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')



data.text = data.text.apply(str)

data.head(5)
#split data in test and train data set

x_train,x_test,y_train,y_test = train_test_split(data['text'], data.sentiment, test_size=0.3, random_state=2020, stratify= data.sentiment)
#Use LogisticRegression Model



pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LogisticRegression())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("LogisticRegression Model accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Get preadiction from model

model.predict(pd.Series(['this is so sad', 'This is good bro!', 'can you help me?']))
#Use LogisticRegression Model



pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model',  LinearSVC())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("LinearSVC accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Get preadiction from model

model.predict(pd.Series(['this is so sad', 'This is good bro!', 'can you help me?']))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', BernoulliNB())])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("BernoulliNB accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', GradientBoostingClassifier(loss = 'deviance',

                                                   learning_rate = 0.01,

                                                   n_estimators = 5,

                                                   max_depth = 500,

                                                   random_state=55))])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("GradientBoostingClassifier accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
#Use DecisionTreeClassifier



pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model',  DecisionTreeClassifier(max_depth=150))])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)

print("DecisionTreeClassifier accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
 # We got max accuracy of 69.07% with LogisticRegression Model