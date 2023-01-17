# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainDf = pd.read_csv("../input/train.csv")

print(trainDf.info())
testDf = pd.read_csv("../input/test.csv")

print(testDf.info())
gender_submDf = pd.read_csv("../input/gender_submission.csv")

print(gender_submDf.info())
print("Missing data in 'Cabin': {:.2f}%".format(trainDf["Cabin"].isnull().sum()/len(trainDf.index)*100.0))

print("Missing data in 'Age': {:.2f}%".format(trainDf["Age"].isnull().sum()/len(trainDf.index)*100.0))
trainDf.drop("Cabin", axis=1, inplace=True)
trainDf["IsFemale"] = trainDf["Sex"].astype("category").cat.codes

trainDf["EmbarkPort"] = trainDf["Embarked"].astype("category").cat.codes

trainDf.drop("Sex", axis=1, inplace=True)

trainDf.drop("Embarked", axis=1, inplace=True)

trainDf.drop(['Name', 'Ticket'], axis=1, inplace=True)

trainDf.head()
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(penalty='l2')

## 'Age' contains missing values, skip for now and see how good we get without it.

LogReg.fit(trainDf.drop(['Survived', 'PassengerId', 'Age'], axis=1), trainDf.Survived)
## It looks like this gives a notion on how well the learned logistic regression predicts the training data

LogReg.score(trainDf.drop(['Survived', 'PassengerId', 'Age'], axis=1), trainDf.Survived)
## Remove the rows with missing values

trainDfAge = trainDf.dropna(axis=0, inplace=False)

trainDfAge.info()

LogRegAge = LogisticRegression(penalty='l2')

LogRegAge.fit(trainDfAge.drop(['Survived', 'PassengerId'], axis=1), trainDfAge.Survived)

## It looks like this gives a notion on how well the learned logistic regression predicts the training data

LogRegAge.score(trainDfAge.drop(['Survived', 'PassengerId'], axis=1), trainDfAge.Survived)
testDf.drop("Cabin", axis=1, inplace=True)

testDf["IsFemale"] = testDf["Sex"].astype("category").cat.codes

testDf["EmbarkPort"] = testDf["Embarked"].astype("category").cat.codes

testDf.drop("Sex", axis=1, inplace=True)

testDf.drop("Embarked", axis=1, inplace=True)

testDf.drop(['Name', 'Ticket', 'Age'], axis=1, inplace=True)

testDf.info()

testDf.head()
testDf1 = testDf.dropna(axis=0, inplace=False)

testDf1.info()
Predict = LogReg.predict(testDf1.drop('PassengerId', axis=1, inplace=False))
answer = pd.DataFrame(Predict, columns=["Survived"])

answer.info()
## Since one row was dropped earlier, I need to reset the index before I add the 'Survived' column, since otherwise the rows (indexes) of the two DataFrames don't line up.

answerDf = pd.concat([testDf1.reset_index(drop=True), answer], axis=1)

answerDf.info()
## Find the affected passenger

noFare = testDf[testDf['Fare'].isnull()]

## Append the data

answerDf = answerDf.append(noFare, sort=False)

## And fill the missing values with 0

answerDf = answerDf.fillna(0)
answerDf[['PassengerId', 'Survived']].to_csv("output.csv", index=False)