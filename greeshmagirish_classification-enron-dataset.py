import numpy as np

import pandas as pd

import os

import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, f1_score
data = pd.read_csv("../input/spam_ham_dataset.csv", header=0)

data = data.drop('Unnamed: 0', axis=1)

data = data.drop('label_num', axis=1)

data.head()

data.describe()
pipeline = Pipeline([

    ('counts', CountVectorizer(ngram_range=(1,2))),

    ('nb', MultinomialNB())

])



pipeline1 = Pipeline([

    ('tfid', TfidfVectorizer()),

    ('lr', LogisticRegression())

])



pipeline2 = Pipeline([

    ('counts', CountVectorizer(ngram_range=(1,2))),

    ('cnb', ComplementNB())

])
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'],test_size=.20)
pipeline.fit(x_train, y_train)
print(classification_report(y_test, pipeline.predict(x_test)))
confusion_matrix(y_test, pipeline.predict(x_test))
pipeline1.fit(x_train, y_train)

print(classification_report(y_test, pipeline1.predict(x_test)))
confusion_matrix(y_test, pipeline1.predict(x_test))
pipeline2.fit(x_train, y_train)

print(classification_report(y_test, pipeline2.predict(x_test)))
confusion_matrix(y_test, pipeline2.predict(x_test))
k_fold = KFold(n_splits=6)

scores = []

confusion = np.array([[0, 0], [0, 0]])

for train_indices, test_indices in k_fold.split(data):

    train_text = data.iloc[train_indices]['text'].values

    train_y = data.iloc[train_indices]['label'].values



    test_text = data.iloc[test_indices]['text'].values

    test_y = data.iloc[test_indices]['label'].values



    pipeline.fit(train_text, train_y)

    predictions = pipeline.predict(test_text)



    confusion += confusion_matrix(test_y, predictions)

    score = f1_score(test_y, predictions, pos_label='spam')

    scores.append(score)



print('Total emails classified:', len(data))

print('Score:', sum(scores)/len(scores))

print('Confusion matrix:')

print(confusion)
k_fold = KFold(n_splits=6)

scores = []

confusion = np.array([[0, 0], [0, 0]])

for train_indices, test_indices in k_fold.split(data):

    train_text = data.iloc[train_indices]['text'].values

    train_y = data.iloc[train_indices]['label'].values



    test_text = data.iloc[test_indices]['text'].values

    test_y = data.iloc[test_indices]['label'].values



    pipeline1.fit(train_text, train_y)

    predictions = pipeline1.predict(test_text)



    confusion += confusion_matrix(test_y, predictions)

    score = f1_score(test_y, predictions, pos_label='spam')

    scores.append(score)



print('Total emails classified:', len(data))

print('Score:', sum(scores)/len(scores))

print('Confusion matrix:')

print(confusion)
k_fold = KFold(n_splits=6)

scores = []

confusion = np.array([[0, 0], [0, 0]])

for train_indices, test_indices in k_fold.split(data):

    train_text = data.iloc[train_indices]['text'].values

    train_y = data.iloc[train_indices]['label'].values



    test_text = data.iloc[test_indices]['text'].values

    test_y = data.iloc[test_indices]['label'].values



    pipeline2.fit(train_text, train_y)

    predictions = pipeline2.predict(test_text)



    confusion += confusion_matrix(test_y, predictions)

    score = f1_score(test_y, predictions, pos_label='spam')

    scores.append(score)



print('Total emails classified:', len(data))

print('Score:', sum(scores)/len(scores))

print('Confusion matrix:')

print(confusion)