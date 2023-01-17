# Importing necessary libraries

import os

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import seaborn as sns

import matplotlib as plt

# Reading Train and Test Data

X = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Viewing and understanding the dataset

X.describe()

X.shape

X.info()

X.head()

X.tail()
# Store the target in another variable

y = X.pop('Survived')

y.head()
# Store the umeric columns for using in the model

num_columns = list(X.dtypes[X.dtypes != 'object'].index)

X[num_columns].head()
# Creating heatmap to understand the relation

sns.heatmap(X[num_columns].corr(),linewidths=0.1,vmax=1.0, 

            square=True, linecolor='white', annot=True)
# Checking for nulls in dataset

X.isnull().sum()
# Substituing the nulls in the age column with mean

X.loc[X.Age.isnull(),'Age'] = X[~X.Age.isnull()].Age.mean()
X[num_columns].head()

X[num_columns].tail()

#X.Cabin.value_counts()

X[num_columns].isnull().sum()
# Building and trining the RandomForestClassifier model

model = RandomForestClassifier(max_depth = 30,n_estimators=100)

model.fit(X[num_columns],y)

accuracy_score(y , model.predict(X[num_columns]))
# Cleaning our test dataset and running the model on it

test.loc[test.Age.isnull(),'Age'] = test[~test.Age.isnull()].Age.mean()

test.loc[test.Fare.isnull(),'Fare'] = 35.62

pred = model.predict(test[num_columns])

pred.sum()
#Storing the prediction in csv

submit = pd.DataFrame({

    "PassengerId" : test["PassengerId"],

    "Survived" : pred

})

submit.to_csv('Prediction.csv',index=False)