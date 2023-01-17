# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import csv

import collections

import os

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

dataset_train = pd.read_csv('../input/train.csv')

dataset_test = pd.read_csv('../input/test.csv')
# Get info on training dataset

dataset_train.info()
# Get info on testing dataset

dataset_test.info()
# checking the column names & sample data for dataset_train

dataset_train.head(10)
# Passenger ID 

dataset_train.PassengerId.describe()
# Data Exploration - Survived

dataset_train.Survived.describe()
dataset_train["Survived"].value_counts()
dataset_train["Survived"].value_counts(normalize = True)
# Data Exploration - Pclass

dataset_train["Pclass"].value_counts(normalize = True)
dataset_train["Survived"].groupby(dataset_train["Pclass"]).mean()
# Data exploration - Name

dataset_train["Name"].describe()
dataset_train["Name"].head(20)
# Data exploration - Sex

dataset_train["Survived"].groupby(dataset_train["Sex"]).mean()
# Data exploration Age

dataset_train["Survived"].groupby(dataset_train["Age"].isnull()).count()
# Check if null age impacts survival rate

dataset_train["Survived"].groupby(dataset_train["Age"].isnull()).mean()
# Data exploration - SibSp

dataset_train["SibSp"].value_counts()
dataset_train["Survived"].groupby(dataset_train["SibSp"]).mean()
# Data exploration - Parch

dataset_train["Parch"].value_counts()
dataset_train["Survived"].groupby(dataset_train["Parch"]).mean()
# Data exploration - Ticket

dataset_train["Ticket"].value_counts()
# Data exploration - Fare

dataset_train.head()
# Fare pentiles

pd.qcut(dataset_train["Fare"], 5).value_counts(sort = False)
# Checking fare correlation with passenger class

pd.crosstab(pd.qcut(dataset_train["Fare"], 5), columns = dataset_train["Pclass"] )
# Checking fare correlation with port of embarkment

pd.crosstab(pd.qcut(dataset_train["Fare"], 4), columns = dataset_train["Embarked"]).apply(lambda r: r/r.sum(), axis=1)
# Correlation of ticket fare with survival rate

dataset_train["Survived"].groupby(pd.qcut(dataset_train["Fare"], 5)).mean()
# Data exploration - Cabin

dataset_train["Cabin"].describe()
dataset_train["Cabin"].head()
# Frequency count of Cabin values

dataset_train["Cabin"].value_counts()
# Checking if null cabin impacts survival

dataset_train["Survived"].groupby(dataset_train["Cabin"].isnull()).mean()
# Data exploration - Embarked

dataset_train["Embarked"].describe()
# Checking the ticket of passengers with empty embarkment

print(dataset_train.loc[dataset_train["Embarked"].isnull()])
# checking correlation between class and port

pd.crosstab(dataset_train["Pclass"], columns = dataset_train["Embarked"]).apply(lambda r: r/r.sum(), axis=1)
# Deleting the passengerID

del dataset_train["PassengerId"]

del dataset_test["PassengerId"]
# Function for extracting titles and removing the Name Column

def titles(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        i["Title"] = i["Name"].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

        del i["Name"]

    return dataset_train, dataset_test
# Function for removing the low incidence titles and bucketing them in to others

def titleGroups(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        i.loc[i["Title"] == "Col.",["Title"]] = "Other" 

        i.loc[i["Title"] == "Major.",["Title"]] = "Other" 

        i.loc[i["Title"] == "Mlle.",["Title"]] = "Other" 

        i.loc[i["Title"] == "Ms.",["Title"]] = "Miss." 

        i.loc[i["Title"] == "Sir.",["Title"]] = "Mr." 

        i.loc[i["Title"] == "Capt.",["Title"]] = "Other" 

        i.loc[i["Title"] == "Lady.",["Title"]] = "Mrs." 

        i.loc[i["Title"] == "Don.",["Title"]] = "Other" 

        i.loc[i["Title"] == "the",["Title"]] = "Other" 

        i.loc[i["Title"] == "Mme.",["Title"]] = "Other" 

        i.loc[i["Title"] == "Jonkheer.",["Title"]] = "Other" 

    return dataset_train, dataset_test
# Function to fill missing age values in the dataset

def fillAges(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        data = dataset_train.groupby(['Title', 'Pclass'])['Age']

        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))

    return dataset_train, dataset_test
# Function to convert siblings and parch to family size

def familySize(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        i["FamilySize"] = np.where((i["SibSp"]+i["Parch"]) == 0 , "Single", np.where((i["SibSp"]+i["Parch"]) <= 3,"Small", "Big"))

        del i["SibSp"]

        del i["Parch"]

    return dataset_train, dataset_test
# Function to append ticketCounts to dataset & delete ticket

def ticketCounts(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        i["TicketCount"] = i.groupby(["Ticket"])["Title"].transform("count")

        del i["Ticket"]

    return dataset_train, dataset_test
# Fill the na Fares with mean of fares from the set.

dataset_train['Fare'].fillna(dataset_train['Fare'].mean(), inplace = True)

dataset_test['Fare'].fillna(dataset_train['Fare'].mean(), inplace = True)
# Function to add Cabin count flag

def cabinCount(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        i["CabinCount"] = i.groupby(["Cabin"])["Title"].transform("count")

        del i["Cabin"]

    return dataset_train, dataset_test
# Function to convert cabinCount to cabinType Flag

def cabinCountFlag(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        i["CabinType"] = "Missing"

        i.loc[i["CabinCount"] == 1,["CabinType"]] = "Single" 

        i.loc[i["CabinCount"] == 2,["CabinType"]] = "Double" 

        i.loc[i["CabinCount"] >= 3,["CabinType"]] = "ThreePlus" 

        del i["CabinCount"]

    return dataset_train, dataset_test 
# Function to fill the missing values of Embarked

def fillEmbarked(dataset_train, dataset_test):

    for i in [dataset_train, dataset_test]:

        i["Embarked"] = i["Embarked"].fillna("S")

    return dataset_train, dataset_test
# Encoding our categorical variables as dummy variables to ensure scikit learn works

def dummies(dataset_train, dataset_test, columns = ["Pclass", "Sex", "Embarked","Title","TicketCount","CabinType","FamilySize"]):

    for column in columns:

        dataset_train[column] = dataset_train[column].apply(lambda x: str(x))

        dataset_test[column] = dataset_test[column].apply(lambda x: str(x))

        good_cols = [column+'_'+i for i in dataset_train[column].unique() if i in dataset_test[column].unique()]

        dataset_train = pd.concat((dataset_train, pd.get_dummies(dataset_train[column], prefix = column)[good_cols]), axis = 1)

        dataset_test = pd.concat((dataset_test, pd.get_dummies(dataset_test[column], prefix = column)[good_cols]), axis = 1)

        del dataset_train[column]

        del dataset_test[column]

    return dataset_train, dataset_test
dataset_train, dataset_test = titles(dataset_train, dataset_test)
dataset_train, dataset_test = titleGroups(dataset_train, dataset_test)
dataset_train, dataset_test = fillAges(dataset_train, dataset_test)
dataset_train, dataset_test = familySize(dataset_train, dataset_test)
dataset_train, dataset_test = ticketCounts(dataset_train, dataset_test)
dataset_train, dataset_test = cabinCount(dataset_train, dataset_test)
dataset_train, dataset_test = cabinCountFlag(dataset_train, dataset_test)
dataset_train, dataset_test = fillEmbarked(dataset_train, dataset_test)
dataset_train, dataset_test = dummies(dataset_train, dataset_test,columns = ["Pclass", "Sex", "Embarked","Title","TicketCount","CabinType", "FamilySize"])
dataset_train.head()
dataset_train.describe()
dataset_test.head()
dataset_test.describe()
# Fitting Random Forest Classification to the Training set

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(max_features='auto', 

                                oob_score=True,

                                random_state=1,

                                n_jobs=-1)
# Creating the Grid Search Parameter list

parameters = { "criterion"   : ["gini", "entropy"],

             "min_samples_leaf" : [1, 5, 10],

             "min_samples_split" : [12, 16, 20, 24],

             "n_estimators": [100, 400, 700]}
# Setting up the gridSearch to find the optimal parameters

gridSearch = GridSearchCV(estimator=classifier,

                  param_grid=parameters,

                  scoring='accuracy',

                  cv=10,

                  n_jobs=-1)
# Getting the optimal grid search parameters

gridSearch = gridSearch.fit(dataset_train.iloc[:, 1:], dataset_train.iloc[:, 0])
# Printing the out of bag score and the best parameters values

print(gridSearch.best_score_)

print(gridSearch.best_params_)
# building the random forrest classifier

classifier = RandomForestClassifier(criterion='entropy', 

                             n_estimators=100,

                             min_samples_split=16,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

classifier.fit(dataset_train.iloc[:, 1:], dataset_train.iloc[:, 0])

print("%.5f" % classifier.oob_score_)
# Creating the list of important features

pd.concat((pd.DataFrame(dataset_train.iloc[:, 1:].columns, columns = ['variable']), 

           pd.DataFrame(classifier.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
# Making the predictions on the test set

predictions = classifier.predict(dataset_test)
# Making the predictions file for submission

predictions = pd.DataFrame(predictions, columns=['Survived'])

passengerIds = pd.read_csv('../input/test.csv')

predictions = pd.concat((passengerIds.iloc[:, 0], predictions), axis = 1)
# To save our results to a csv locally

predictions.to_csv('predictions.csv', index = False)