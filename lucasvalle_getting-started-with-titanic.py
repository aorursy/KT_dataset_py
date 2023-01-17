import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



train.info()

test.info()
train.drop (["PassengerId", "Name", "Cabin", "Ticket"], axis = 1, inplace = True)

test.drop (["Name", "Cabin", "Ticket"], axis = 1, inplace = True)
sns.distplot (train ["Age"])



train ["Age"].fillna (train ["Age"].median(), inplace = True)

test ["Age"].fillna (test ["Age"].median(), inplace = True)
sns.countplot (train ["Embarked"])

train ["Embarked"].fillna ("S", inplace = True)
sns.distplot (test ["Fare"])

test ["Fare"].fillna (test ["Fare"].median(), inplace = True)
#dataset with no missing data

train.info()

test.info()
train ["FamSize"] = train ["SibSp"] + train ["Parch"] + 1

test ["FamSize"] = test ["SibSp"] + test ["Parch"] + 1



train.drop (["SibSp", "Parch"], axis = 1, inplace = True)

test.drop (["SibSp", "Parch"], axis = 1, inplace = True)



sns.countplot (train ["FamSize"], hue = train ["Survived"])
sns.heatmap (train.corr(), annot = True)
from sklearn.linear_model import LogisticRegression



X_train = pd.get_dummies (train.loc[:, train.columns != "Survived"])

y_train = train ["Survived"]



X_test = pd.get_dummies (test.loc [:, test.columns != "PassengerId"])



clf = LogisticRegression().fit (X_train, y_train)

print (clf.score (X_train, y_train))



y_pred = clf.predict (X_test)



submission = pd.DataFrame ({"PassengerId": test ["PassengerId"], "Survived": y_pred})

submission.to_csv ("submission.csv", index = False)