# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head()
cols_to_drop = ['Name', 'Ticket', 'Cabin']

train_df = train_df.drop(cols_to_drop, axis=1)

test_df = test_df.drop(cols_to_drop, axis=1)
train_df = train_df.drop(['PassengerId'], axis=1)

train_df.head()
train_df.describe()
train_df.groupby(['Pclass', 'Sex']).describe()
train_df['Age'] = train_df['Age'].fillna(train_df.groupby(['Sex', 'Pclass'])['Age'].transform('mean'))
train_df = train_df.dropna(axis=0)
test_df.describe()
test_df['Age'] = test_df['Age'].fillna(test_df.groupby(['Sex', 'Pclass'])['Age'].transform('mean'))
train_df.count()
test_df.count()
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_df.head()
train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
model = LogisticRegression()

model.fit(X_train, Y_train)

Y_preds = model.predict(X_test)

model.score(X_train, Y_train)
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": Y_preds})
submission.to_csv('preds.csv', index=False)