#import modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV
#load the train and test data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
#concatenate train and test data for manipulation

data = pd.concat([train_data, test_data])
#filling missing values

data['Age'] = data.Age.fillna(data.Age.median())

data['Fare'] = data.Fare.fillna(data.Fare.median())

data['Embarked'] = data['Embarked'].fillna('S')
#mapping of sex

data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#create a category "Has_Cabin"

data['Has_Cabin'] = ~data.Cabin.isnull()

data['Has_Cabin'] = data['Has_Cabin'].astype(int)
#Extract the "Title" from the feature "Name"

data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})

data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',

                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
#combining the categories "SibSp" and "Parch" to "Family"

data["Family"] = data["SibSp"] + data["Parch"]
#bin the features "Age", "Fare" and "Family"

data['CatAge'] = pd.cut(data.Age,bins=[0,6,30,60,80],labels=["0-6","6-30","30-60","60-80"],include_lowest=True)

data['CatFare']= pd.cut(data.Fare,bins=[0,1,10,100,600],labels=["0-1","1-10","10-100","100-600"],include_lowest=True)

data['CatFamily']= pd.cut(data.Family,bins=[0,0.5,3.5,10],labels=["0", "1-2","3-10"],include_lowest=True)
#drop features that will not be used for the modeling

data = data.drop(["PassengerId", "Name","SibSp","Parch","Ticket","Cabin","Age","Fare","Family"], axis=1)
#create dummy variables

data_dum = pd.get_dummies(data, drop_first=True)
#split the data frame "data" into train and test data

data_train = data_dum.iloc[:891]

data_test = data_dum.iloc[891:].drop("Survived", axis=1)
#split the train data into label and features

y_train = data_train["Survived"]

X_train = data_train.drop("Survived", axis=1)
#select values

X = X_train.values

test = data_test.values

y = y_train.values
#Implement a Random Forest Classifier

clf = RandomForestClassifier()
#Implement a cross-validated grid-search

param_grid = {

                 'n_estimators': [20, 21, 22, 23, 24],

                 'max_depth': [4, 5, 6, 7]

             }



grid_clf = GridSearchCV(clf, param_grid, cv=5)
#fit the grid-search

grid_clf.fit(X,y)
#Print the best parameters of the grid-search

print(grid_clf.best_params_)
#Predict the labels of the test data and create a CSV-file for submission

y_pred = grid_clf.predict(test)

test_data['Survived'] = y_pred

test_data['Survived'] = test_data['Survived'].astype(int)

test_data[['PassengerId', 'Survived']].to_csv('titanic_RandomForest_GridSearch-Final.csv', index=False)

#Kaggle Titanic AUC: 0.80861