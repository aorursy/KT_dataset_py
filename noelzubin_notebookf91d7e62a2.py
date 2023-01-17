%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# backup copy of original data

backup_1 =  train.copy()

backup_2 = test.copy()



# calculate family size

train["FamilySize"] = train.SibSp + train.Parch + 1

test["FamilySize"] = test.SibSp + test.Parch + 1



# vectorize Embarked

cat = pd.Categorical(train.Embarked)

train["Embarked"] = cat.codes

test["Embarked"] = pd.Categorical(test.Embarked, categories=cat.categories, ordered=True ).codes
# to fill the missing Embarked Value from Fare and Pclass using KNN.

neigh = KNeighborsClassifier(n_neighbors=2)

temp =  backup_1[["Fare", "Pclass", "Embarked"]].dropna()

temp["Embarked"] = pd.Categorical(temp.Embarked, categories=cat.categories, ordered=True).codes

neigh.fit(temp[["Fare", "Pclass"]], temp["Embarked"])

def get_missing_embarked(fare, pclass):

    return neigh.predict([fare, pclass])

# fill the empty embarked

for i in backup_1.loc[pd.isnull(backup_1["Embarked"])].index:

    train["Embarked"][i] = get_missing_embarked(train.Fare[i], train.Pclass[i])

for i in backup_2.loc[pd.isnull(backup_2["Embarked"])].index:

    test["Embarked"][i] = get_missing_embarked(test.Fare[i], test.Pclass[i])   



# remove useless data

train.drop(["SibSp", "Parch", "Ticket"], axis=1, inplace=True)

test.drop(["SibSp", "Parch", "Ticket"], axis=1, inplace=True)



# get the Designation from the name and delete the name column( Mr, Mrs ...)

train["Desig"] = train.apply( lambda x: re.split("[,.]", x["Name"])[1].strip() ,axis=1)

test["Desig"] = test.apply( lambda x: re.split("[,.]", x["Name"])[1].strip() ,axis=1)

del train["Name"]

del test["Name"]



# Change Designations to vectors.

cat = pd.Categorical(train["Desig"])

train["Desig"] = cat.codes

test["Desig"] = pd.Categorical(test["Desig"], categories=cat.categories, ordered=True).codes



# Add Binary vector to show if the person has a cabin

train["Cabin"][pd.notnull(train["Cabin"])] = 1

train["Cabin"][pd.isnull(train["Cabin"])] = 0

test["Cabin"][pd.notnull(test["Cabin"])] = 1

test["Cabin"][pd.isnull(test["Cabin"])] = 0

del train["Cabin"]

del test["Cabin"]



# fix missing fare with mean

train.Fare = train.Fare.fillna(train.Fare.mean())

test.Fare = test.Fare.fillna(test.Fare.mean())



# fill missing age with the average value for each designation

fill_values = train[["Age","Desig"]].apply(lambda x: backup_1[train["Desig"] == x[1]].Age.mean(), axis= 1)

train["Age"] = train["Age"].fillna(fill_values)

fill_values = test[["Age","Desig"]].apply(lambda x: backup_2[test["Desig"] == x[1]].Age.mean(), axis= 1)

test["Age"] = test["Age"].fillna(fill_values)

test["Age"] = test["Age"].fillna(test["Age"].mean())



# vectorize Sex

cat = pd.Categorical(train.Sex)

train["Sex"] = cat.codes

test["Sex"] = pd.Categorical(test.Sex, categories=cat.categories, ordered=True).codes



trainX = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Desig']]

trainY = train["Survived"]

testX = test[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Desig']]



# Ensemble Learn

def getEnsemble(trainX, trainY, testX):

    clf1 = LogisticRegression()

    clf2 = RandomForestClassifier()

    clf3 = ExtraTreesClassifier()

    clf4 = KNeighborsClassifier(n_neighbors=1)

    model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('knn', clf4)], voting='soft')

    model.fit(trainX, trainY)

    print(model.score(trainX, trainY))

    return model.predict(testX)



result = pd.DataFrame ( test["PassengerId"] )

result["Survived"] = getEnsemble(trainX, trainY, testX)

result.to_csv("Results.csv", index=None)