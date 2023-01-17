import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train_df.info()
train_df.head()
# check the class distribution for the target label in train_df?

train_df['target'].value_counts()
X = train_df['text']

y = train_df['target']
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
# examine the class distribution in y_train and y_test

print(y_train.value_counts(),'\n', y_val.value_counts())
# import and instantiate CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vect = CountVectorizer(lowercase=True, stop_words='english', token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')

vect
# learn the vocabulary in the training data, then use it to create a document-term matrix

X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix created from X_train

X_train_dtm
# transform the test data using the earlier fitted vocabulary, into a document-term matrix

X_val_dtm = vect.transform(X_val)

# examine the document-term matrix from X_test

X_val_dtm
# import and instantiate the Logistic Regression model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=8)

logreg
# tune hyperparameter

from sklearn.model_selection import GridSearchCV

grid_values = {'C':[0.01, 0.1, 1.0, 3.0, 5.0]}

grid_logreg = GridSearchCV(logreg, param_grid=grid_values, scoring='neg_log_loss', cv=5)

grid_logreg.fit(X_train_dtm, y_train)

grid_logreg.best_params_
# set with recommended parameter

logreg = LogisticRegression(C=1.0, random_state=8)

# train the model using X_train_dtm & y_train

logreg.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm

y_pred_val = logreg.predict(X_val_dtm)
# compute the accuracy of the predictions

from sklearn import metrics

metrics.accuracy_score(y_val, y_pred_val)
# compute the accuracy of predictions with the training data

y_pred_train = logreg.predict(X_train_dtm)

metrics.accuracy_score(y_train, y_pred_train)
# look at the confusion matrix for y_test

metrics.confusion_matrix(y_val, y_pred_val)
# compute the predicted probabilities for X_test_dtm

y_pred_prob = logreg.predict_proba(X_val_dtm)

y_pred_prob[:10]
# compute the log loss number

metrics.log_loss(y_val, y_pred_prob)
# Learn the vocabulary in the entire training data, and create the document-term matrix

X_dtm = vect.fit_transform(X)

# Examine the document-term matrix created from X_train

X_dtm
# Train the Logistic Regression model using X_dtm & y

logreg.fit(X_dtm, y)
# Compute the accuracy of training data predictions

y_pred_train = logreg.predict(X_dtm)

metrics.accuracy_score(y, y_pred_train)
test = test_df['text']

# transform the test data using the earlier fitted vocabulary, into a document-term matrix

test_dtm = vect.transform(test)

# examine the document-term matrix from X_test

test_dtm
# make author (class) predictions for test_dtm

LR_y_pred = logreg.predict(test_dtm)

print(LR_y_pred)
# calculate predicted probabilities for test_dtm

LR_y_pred_prob = logreg.predict_proba(test_dtm)

LR_y_pred_prob[:10]
submission['target'] = LR_y_pred
submission
# Generate submission file in csv format

submission.to_csv('submission.csv', index=False)