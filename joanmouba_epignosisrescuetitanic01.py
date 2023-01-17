# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

from subprocess import check_output

import matplotlib 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.tools.plotting import scatter_matrix

import seaborn as sns



import sklearn as sk

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Imputer

from sklearn.dummy import DummyClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score



%matplotlib inline



print(check_output(["ls", "../input"]).decode("utf8"))



print("\n### Load data ###")

titanic_train_path = "../input/" + "train.csv"

titanic_test_path = "../input/" + "test.csv"

train = pd.read_csv(titanic_train_path, header=0)

test  = pd.read_csv(titanic_test_path, header=0)

PassengerId = test["PassengerId"]

print("X shape: ", X.shape, "; y shape: ", y.shape)



print("\n### Quick look at the data structure ###")

#train.head()

train.describe()

#train.info()

#train.hist(bins=25, figsize=(20,15))



print("\n### Preparing data for models: preprocessing ###")

print("\n### 1. Clean data ###")

print("fill age with median age, Embarked with S, replace Cabin by Has_Cabin")

median_age = train["Age"].median()

train["Age"] = train["Age"].fillna(median_age)

test["Age"] = test["Age"].fillna(median_age)

train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")

train["Has_Cabin"] = train["Cabin"].apply(lambda x:0 if type(x)==float else 1)

test["Has_Cabin"] = test["Cabin"].apply(lambda x: 0 if type(x)==float else 1)

train.head(3)



print("\n### 2. Transform data ###")

print("encode embarked, Sex; Drop PassengerId, Name, Ticketand Cabin")

encoder = LabelEncoder()

train["Embarked"] = encoder.fit_transform(train["Embarked"])

test["Embarked"] = encoder.fit_transform(test["Embarked"])

train["Sex"] = encoder.fit_transform(train["Sex"])

test["Sex"] = encoder.fit_transform(test["Sex"])

corr_matrix = train.corr()

#print(corr_matrix["Survived"].sort_values(ascending=False))

#attributes = ["Survived", "Pclass", "Has_Cabin", "Sex"]

#scatter_matrix(train[attributes], figsize=(12,8))



columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "Parch", "Age" ]

train = train.drop(columns_to_drop, axis=1)

test  = test.drop(columns_to_drop, axis=1)

train.head(3)

test.head(4)

print("\n### Split data in training and testing sets ###")

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

y_train = train["Survived"].ravel()

train = train.drop(["Survived"], axis=1)

X_train = train.as_matrix() #train.values

X_test = test.as_matrix() #test.values

print("X_train shape: ", X_train.shape, "; X_test shape: ", X_test.shape)

train.head(3)



print("\n### Choose and train a model ###\n")

N_ESTIMATORS = 100

clf_dummy = DummyClassifier(strategy='most_frequent')

clf_svc = SVC()

clf_sgd = SGDClassifier(alpha=0.0001)

clf_rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=6, min_samples_leaf=2)

clf_ext = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=8)

clf_abc = AdaBoostClassifier(n_estimators=N_ESTIMATORS)

clf_gbc = GradientBoostingClassifier(n_estimators=N_ESTIMATORS)

clf_voting = VotingClassifier(estimators=[('svc', clf_svc), ('sgd', clf_sgd), 

                                          ('rf', clf_rf), ('ext', clf_ext), 

                                         ('abc',clf_abc), ('gbc',clf_gbc), ], voting='hard')

clf = clf_dummy

cc = clf.fit(X_train, y_train)



print("\n### Predict ###")

predictions = clf.predict(X_test)



print("\n### Evaluate model ###")

#print("accuracy: ", accuracy_score(y_test_pre, y_pred))



print("\n ### Generate Submission File ###") 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)

# Any results you write to the current directory are saved as output.

train.head(3)

print("\ntest head")

test.head(4)