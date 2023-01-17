import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
# import data train

df = pd.read_csv('../input/train.csv')

display(df.info())

display(df.head())
# drop columns

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis = 1)

df.head()
# dummy sex column

df = pd.get_dummies(df, columns = ['Sex'], drop_first = True)

display(df.head())

len(df)
display(df.corr())

sns.pairplot(df)
# data feature selection

X = df.drop(['Survived', 'Fare'], axis = 1)

y = df['Survived']

X.head()
# logistic regression model

log_reg = LogisticRegression(solver = 'liblinear')

log_reg.fit(X, y)



# scoring model

print('Accuracy = {}%'.format(round(log_reg.score(X, y) * 100, 2)))
# knn model

knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn.fit(X, y)



# scoring model

print('Accuracy = {}%'.format(round(knn.score(X, y) * 100, 2)))
# kernel svm model

svm = SVC(kernel = 'rbf', gamma = 0.5)

svm.fit(X, y)



# scoring model

print('Accuracy = {}%'.format(round(svm.score(X, y) * 100, 2)))
# naive bayes model

gnb = GaussianNB()

gnb.fit(X, y)



# scoring model

print('Accuracy = {}%'.format(round(gnb.score(X, y) * 100, 2)))
# decision tree classifier

dct_clf = DecisionTreeClassifier(criterion = 'entropy')

dct_clf.fit(X, y)



# scoring model

print('Accuracy = {}%'.format(round(dct_clf.score(X, y) * 100, 3)))
# random forest model

rf_clf = RandomForestClassifier(n_estimators = 25, criterion = 'entropy')

rf_clf.fit(X, y)



# scoring model

print('Accuracy = {}%'.format(round(rf_clf.score(X, y) * 100, 3)))
# import data test

df_test = pd.read_csv('../input/test.csv')

display(df_test.info())

display(df_test.head())
# dummy sex column

df_test = pd.get_dummies(df_test, columns = ['Sex'], drop_first = True)

df_test.head()
# data selection

X_test = df_test[['Pclass', 'SibSp', 'Parch', 'Sex_male']]

X_test.head()
# prediction model using the highest accuracy model = random forest

df_test['Survived'] = rf_clf.predict(X_test)

df_test
# save data prediction to csv

df_submit = df_test[['PassengerId', 'Survived']]

df_submit.head()
import csv

df_submit.to_csv('titanic_prediction.csv', index = False)