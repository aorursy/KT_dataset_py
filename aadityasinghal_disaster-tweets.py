import numpy as np

import pandas as pd

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 
dataset_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
dataset_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
corpus_train = []

for i in range(0, 7613):

    text_train = re.sub('[^a-zA-Z]', ' ', dataset_train['text'][i])

    text_train = text_train.lower()

    text_train = text_train.split()

    ps_train = PorterStemmer()

    text_train = [ps_train.stem(word) for word in text_train if not word in set(stopwords.words('english'))]

    text_train = ' '.join(text_train)

    corpus_train.append(text_train)
corpus_test = []

for i in range(0, 3263):

    text_test = re.sub('[^a-zA-Z]', ' ', dataset_test['text'][i])

    text_test = text_test.lower()

    text_test = text_test.split()

    ps_test = PorterStemmer()

    text_test = [ps_test.stem(word) for word in text_test if not word in set(stopwords.words('english'))]

    text_test = ' '.join(text_test)

    corpus_test.append(text_test)
cv = CountVectorizer(max_features = 1840)

X_train = cv.fit_transform(corpus_train).toarray()

X_test = cv.fit_transform(corpus_test).toarray()

y_train = dataset_train.iloc[:, 4].values
from xgboost import XGBClassifier

classifier = XGBClassifier(booster = 'gbtree', gamma = 0.8, max_depth = 30, learning_rate = 0.62,n_estimators=150)

classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)



print('Confusion Matrix :')

print(confusion_matrix(y_train, y_pred_train)) 

print('Accuracy Score :',accuracy_score(y_train, y_pred_train))

print('Report : ')

print(classification_report(y_train, y_pred_train))
y_pred_test = classifier.predict(X_test)

output = pd.DataFrame({'id': dataset_test.id, 'target': y_pred_test})

output.to_csv('my_submission_nlp_30.csv', index=False)

print("Your submission was successfully saved!")