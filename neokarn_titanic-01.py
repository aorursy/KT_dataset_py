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



train_df = pd.read_csv('../input/train.csv')

train_df = train_df.drop(['PassengerId','Name','Ticket', 'Cabin','Fare'], axis=1)



test_df = pd.read_csv('../input/test.csv')

test_df = test_df.drop(['Name','Ticket', 'Cabin','Fare'], axis=1)



combine = [train_df, test_df]



freq_port = train_df.Embarked.dropna().mode()[0]

mean_age = train_df.Age.dropna().mean()



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0.0, 'C': 1.0, 'Q': 2.0}).astype(int)

    dataset['Age'] = dataset['Age'].fillna(mean_age)

    

X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']

X_test  = test_df.drop('PassengerId', axis=1).copy()    
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })



submission.to_csv('gender_submission.csv', index=False)
train_df.info()
train_df.head()
test_df.info()
test_df.head()
submission.head()