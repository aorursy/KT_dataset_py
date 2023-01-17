import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import IPython

from IPython import display

import seaborn as sns 

import sklearn as skl

#ignore annoying warnings

import warnings

warnings.filterwarnings('ignore')
train_file_path = '../input/train.csv'

train = pd.read_csv(train_file_path)

test_file_path = '../input/test.csv'

test = pd.read_csv(test_file_path)
normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}

    

def createTitle(dataframe):

    dataframe['Title'] = dataframe.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

    dataframe.Title = dataframe.Title.map(normalized_titles)



createTitle(train)

createTitle(test)
grouped = train.groupby(["Sex", "Pclass", "Title"])



train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

test.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

train['Sex_Code'] = label.fit_transform(train['Sex'])

test['Sex_Code'] = label.fit_transform(test['Sex'])



train = pd.get_dummies(train, columns=["Title"])

test = pd.get_dummies(test, columns=["Title"])

train = pd.get_dummies(train, columns=["Pclass"])

test = pd.get_dummies(test, columns=["Pclass"])

columns = ["Sex_Code","Age","Title_Miss","Title_Mr","Title_Mrs","Title_Officer","Title_Royalty","Pclass_2","Pclass_3"]

y_train = train["Survived"]



train = train[columns]

X_test = test[columns]



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



scaler.fit(train.values)

scaled_features = scaler.transform(train.values)

scaled_features_train = pd.DataFrame(scaled_features, index=train.index, columns=train.columns)



scaled_features = scaler.transform(X_test.values)

scaled_features_test = pd.DataFrame(scaled_features, index=X_test.index, columns=X_test.columns)



#We will not use test anymore
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
# Logistic Regression

lr = LogisticRegression()

cv_results = cross_validate(lr, train, y_train, cv=10, return_train_score=False)

print(cv_results["test_score"].mean())
# Support Vector Machines

svc = SVC()

cv_results = cross_validate(svc, train, y_train, cv=10, return_train_score=False)

print(cv_results["test_score"].mean())
#K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors = 5)

cv_results = cross_validate(knn, train, y_train, cv=10, return_train_score=False)

print(cv_results["test_score"].mean())
# Perceptron

perceptron = Perceptron()

cv_results = cross_validate(perceptron, train, y_train, cv=10, return_train_score=False)

print(cv_results["test_score"].mean())
# Decision Tree

decision_tree = DecisionTreeClassifier()

cv_results = cross_validate(decision_tree, train, y_train, cv=10, return_train_score=False)

print(cv_results["test_score"].mean())
# Random Forest

random_forest = RandomForestClassifier(n_estimators=50)

cv_results = cross_validate(decision_tree, train, y_train, cv=10, return_train_score=False)

print(cv_results["test_score"].mean())
#Fit random forest and make the predictions over the test

lr.fit(train, y_train)

Y_pred = lr.predict(X_test)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })



#submission.to_csv('../output/submission.csv', index=False)