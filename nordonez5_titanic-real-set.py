import os

print(os.listdir("../input"))

import numpy as np

import pandas as pd 

import math as ma 

import matplotlib.pyplot as plt

import seaborn as sn

import sklearn as sk
testset = pd.read_csv("../input/titanic3.csv")
testset.info()
sn.countplot(x ="survived", data = testset)
sn.countplot(x = "survived", hue = "pclass", data = testset )
sn.countplot(x = 'survived', hue = "sibsp", data = testset)
sn.scatterplot(x = 'fare', y = 'survived', data = testset)
testset.drop("cabin", axis=1, inplace=True)
testset.head(2)
testset.isnull().sum()
sn.heatmap(testset.isnull(), yticklabels=False, cbar=True)
sex = pd.get_dummies(testset['sex'], drop_first=True)
embarked = pd.get_dummies(testset["embarked"], drop_first=True)

pcl = pd.get_dummies(testset["pclass"], drop_first=True)
sibsp = pd.get_dummies(testset['sibsp'])
titanic_data = pd.concat([testset, sex, embarked, pcl, sibsp], axis=1)
titanic_data.head(20)
titanic_data.drop(["sex", "embarked", "parch", "name", "ticket", 'boat', 'home.dest',], axis=1,inplace=True)
titanic_data.drop(['body'], axis =1, inplace=True)
titanic_data.head(2)
titanic_data.drop(['fare', 'age'], axis =1, inplace=True)
sn.heatmap(titanic_data.isnull(), yticklabels=False, cbar=True)
#Training Data

X = titanic_data.drop("survived", axis=1)

y = titanic_data["survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
##Testing accuracy

titanic_data2 = pd.concat([testset, sex], axis=1)
titanic_data2.head(2)
titanic_data2.drop(["sex", "embarked", "parch", "name", "ticket", 'boat', 'home.dest'], axis=1,inplace=True)
titanic_data2.drop(['age', 'fare', 'body', 'sibsp'], axis = 1, inplace=True )
titanic_data2.head(2)
##Train Data 2

X = titanic_data2.drop("survived", axis=1)

y = titanic_data2["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
classification_report(y_test,predictions)
confusion_matrix(y_test, predictions)
accuracy_score(y_test, predictions)