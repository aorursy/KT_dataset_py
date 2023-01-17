import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import csv as csv

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.isnull().sum()
test.isnull().sum()
train2= pd.read_csv("../input/train.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)

test2= pd.read_csv("../input/test.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test2["Fare"].fillna(test.Fare.median(), inplace=True)
train2["Family_size"] = train2["SibSp"] + train2["Parch"] +1 

test2["Family_size"] = test2["SibSp"] + test2["Parch"] +1
train2["Alone"] = 0

train2.loc[train2["Family_size"] == 1, "Alone"] = 1

test2["Alone"] = 0

test2.loc[test2["Family_size"] == 1, "Alone"] = 1
for train2 in [train2]:

        train2['Title'] = train2.Name.str.extract(' ([A-Za-z]+).', expand=False)

for train2 in [train2]:

        train2['Title'] = train2['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

        train2['Title'] = train2['Title'].replace('Mlle', 'Miss')

        train2['Title'] = train2['Title'].replace('Ms', 'Miss')

        train2['Title'] = train2['Title'].replace('Mme', 'Mrs')

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5} 



for train2 in [train2]: 

        train2['Title'] = train2['Title'].map(Salutation_mapping) 

        train2['Title'] = train2['Title'].fillna(0)
for test2 in [test2]:

        test2['Title'] = test2.Name.str.extract('([A-Za-z]+).', expand=False) 

for test2 in [test2]:

        test2['Title'] = test2['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

        test2['Title'] = test2['Title'].replace('Mlle', 'Miss')

        test2['Title'] = test2['Title'].replace('Ms', 'Miss')

        test2['Title'] = test2['Title'].replace('Mme', 'Mrs')

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}



for test2 in [test2]:

        test2['Title'] = test2['Title'].map(Salutation_mapping)

        test2['Title'] = test2['Title'].fillna(0)
train2.loc[(train2['Name'].str.contains('Mr\.')) & (train2['Age'].isnull()), 'Age'] = train2[train2['Name'].str.contains('Mr\.')].Age.mean()

train2.loc[(train2['Name'].str.contains('Mrs\.')) & (train2['Age'].isnull()), 'Age'] = train2[train2['Name'].str.contains('Mrs\.')].Age.mean()

train2.loc[(train2['Name'].str.contains('Miss\.')) & (train2['Age'].isnull()), 'Age'] = train2[train2['Name'].str.contains('Miss\.')].Age.mean()

train2.loc[(train2['Name'].str.contains('Master\.')) & (train2['Age'].isnull()), 'Age'] = train2[train2['Name'].str.contains('Master\.')].Age.mean()

train2.loc[(train2['Name'].str.contains('Other\.')) & (train2['Age'].isnull()), 'Age'] = train2[train2['Name'].str.contains('Other\.')].Age.mean()
test2.loc[(test2['Name'].str.contains('Mr\.')) & (test2['Age'].isnull()), 'Age'] = test2[test2['Name'].str.contains('Mr\.')].Age.mean()

test2.loc[(test2['Name'].str.contains('Mrs\.')) & (test2['Age'].isnull()), 'Age'] = test2[test2['Name'].str.contains('Mrs\.')].Age.mean()

test2.loc[(test2['Name'].str.contains('Miss\.')) & (test2['Age'].isnull()), 'Age'] = test2[test2['Name'].str.contains('Miss\.')].Age.mean()

test2.loc[(test2['Name'].str.contains('Master\.')) & (test2['Age'].isnull()), 'Age'] = test2[test2['Name'].str.contains('Master\.')].Age.mean()

test2.loc[(test2['Name'].str.contains('Other\.')) & (test2['Age'].isnull()), 'Age'] = test2[test2['Name'].str.contains('Other\.')].Age.mean()
import xgboost as xgb

from xgboost import XGBClassifier



target = train2["Survived"].values

features = train2[["Pclass","Age","Sex","Fare","Family_size","Alone"]].values

model = XGBClassifier(n_estimators=30, random_state=9)

model = model.fit(features, target)



test_features = test2[["Pclass","Age","Sex","Fare","Family_size","Alone"]].values

prediction = model.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)

output = pd.DataFrame(prediction, PassengerId, columns=["Survived"])

output.to_csv("submit5-76.csv", index_label=["PassengerId"])