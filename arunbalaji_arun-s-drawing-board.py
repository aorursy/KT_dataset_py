

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

# Import RandomForestClassifier from sklearn

from sklearn.ensemble import RandomForestClassifier



# Import train.csv & test.csv

trainCsv = pd.read_csv('/kaggle/input/titanic/train.csv')

testCsv = pd.read_csv('/kaggle/input/titanic/test.csv')



# Gather elibible column names from the givern tain set

eligible_columns = ["Pclass", "Sex", "Embarked", "SibSp"]



# Extract eligible_columns from train set

X = pd.get_dummies(trainCsv[eligible_columns])



# Extract Survived details from train.csv

Y = trainCsv["Survived"]



# Import and create a classifier

clfModel = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1)

clfModel.fit(X, Y)



# Extract eligible_columns from test set

Z = pd.get_dummies(testCsv[eligible_columns])



# Gather Prediction set

RFCResult = clfModel.predict(Z)



# Extract result

result = pd.DataFrame({'PassengerId': testCsv.PassengerId, 'Survived': RFCResult})



#Write to a csv file

result.to_csv('predicted_surviors.csv', index=False)






