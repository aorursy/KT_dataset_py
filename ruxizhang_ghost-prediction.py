import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LogisticRegression
train = pd.read_csv("../input/train.csv", index_col=0)

test = pd.read_csv("../input/test.csv", index_col=0)

test.head()
train.head()
##create dummies from categorical variables

color_train = pd.get_dummies(train['color'])

color_test = pd.get_dummies(test['color'])

train_new = pd.concat([train, color_train], axis=1)

test_new = pd.concat([test, color_test], axis=1)

train_new.drop(['color','type'],inplace=True,axis=1)

test_new.drop(['color'],inplace=True,axis=1)

test_new.head()
##create interaction term matrix

train_Xs = pd.DataFrame(PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(train_new))

train_Y = pd.Series(train['type'])

test_Xs = pd.DataFrame(PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(test_new))

train_Xs.head()
##fit a simple logistic regression

lr = LogisticRegression()

lr.fit(train_Xs, train_Y)

# make predictions

expected = train_Y

predicted = lr.predict(train_Xs)

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))