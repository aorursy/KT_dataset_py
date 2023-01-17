## Import useful libraries

import numpy as np # linear algrebra

import pandas as pd # data processing

from pandas import Series,DataFrame

import numpy as np 

import matplotlib.pyplot as plt # plots

import seaborn as sns  # statistical data visualization



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC
## Get data frame

titanic_test = pd.read_csv("../input/test.csv")

titanic_train = pd.read_csv("../input/train.csv")



# Preview & Describe

print(titanic_train.head())

print(titanic_train.describe())

print(titanic_train.info())
## Convert Sex into a bolean column

titanic_test.loc[titanic_test['Sex'] == 'female','Sex'] = 1

titanic_test.loc[titanic_test['Sex'] == 'male','Sex'] = 0

titanic_train.loc[titanic_train['Sex'] == 'female','Sex'] = 1

titanic_train.loc[titanic_train['Sex'] == 'male','Sex'] = 0
## Embarked column

# Visualization

sns.countplot('Embarked', data = titanic_train)

sns.factorplot('Embarked','Survived', data = titanic_train)



# Replace missing values with S

titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")



# Convert into numeric values

titanic_train.loc[titanic_train["Embarked"] == "S","Embarked"] = 0

titanic_train.loc[titanic_train["Embarked"] == "C","Embarked"] = 1

titanic_train.loc[titanic_train["Embarked"] == "Q","Embarked"] = 2



titanic_test.loc[titanic_test["Embarked"] == "S","Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C","Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q","Embarked"] = 2
## Fare

print(titanic_train["Fare"].describe()) # no missing values

 

# Visualize 

titanic
## Family

# Create one column for family 