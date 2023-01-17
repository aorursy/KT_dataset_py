##Lets import Packages and Read data..



##analytics packages

import pandas as pd

import numpy as np



##visualization packages

import matplotlib.pyplot as plt



##reading data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
##Lets take a look at our data.

train.describe(include ="all")
##Damn.. what are the values that have 'NULL'?

pd.isnull(train).sum()
##Filling the null values of 'Age'!

age_mean=train['Age'].mean()

train['Age']=train['Age'].fillna(age_mean)



##Time to fill the null values of 'Embarked'

##for that we cant simply fill anything..so lets see where majority of people embarked from..



southampton = train[train["Embarked"] == "S"].shape[0]

cherbourg = train[train["Embarked"] == "C"].shape[0]

queenstown = train[train["Embarked"] == "Q"].shape[0]



print("No. of people from Southampton (S) = ",southampton)

print("No. of people from Cherbourg   (C) = ",cherbourg)

print("No. of people from Queenstown  (Q) = ",queenstown)

##now that we see majority of people are from Southampton.. so we replace embarked with (S)..

train["Embarked"] = train["Embarked"].fillna("S")
##time to fill null values in 'Fare'..



fare_median=train["Fare"].median()

train["Fare"] = train["Fare"].fillna(fare_median)
pd.isnull(train).sum()
fare_means = train.pivot_table('Fare', index='Pclass', aggfunc='mean')



test['Fare'] = test[['Fare', 'Pclass']].apply(lambda x:

                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])

                            else x['Fare'], axis=1)

test['Gender'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)

test = pd.concat([test, pd.get_dummies(test['Embarked'], prefix='Embarked')],

                axis=1)



test = test.drop(['Sex', 'Embarked'], axis=1)



test_data = test.values



output = model.predict(test_data[:,1:])