import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
y = train['Survived'] #select the column representing survival 

X = train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True) # drop the irrelevant columns and keep the rest

X = pd.get_dummies(train) # convert non-numerical variables to dummy variables
from sklearn import tree

dtc = tree.DecisionTreeClassifier()

dtc.fit(X, y)
test = pd.read_csv("../input/titanic/test.csv") # load the testing data

ids = test[['PassengerId']] # create a sub-dataset for submission file and saving it

test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True) # drop the irrelevant and keeping the rest

test.fillna(2, inplace=True) # fill (instead of drop) empty rows so that I would get the exact row number required for submission

test = pd.get_dummies(test) # convert non-numerical variables to dummy variables
predictions = dtc.predict(test)
results = ids.assign(Survived = predictions) # assign predictions to ids

results.to_csv("titanic-results.csv", index=False) # write the final dataset to a csv file.