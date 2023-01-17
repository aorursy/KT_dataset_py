# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import math

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
trainDf.drop(['Name', 'Ticket'], axis=1, inplace=True)
trainDf["IsFemale"] = trainDf["Sex"].astype("category").cat.codes

trainDf.drop("Sex", axis=1, inplace=True)

trainDf["EmbarkPort"] = trainDf["Embarked"].astype("category").cat.codes

trainDf.drop("Embarked", axis=1, inplace=True)

trainDf['CabinNo'] = trainDf["Cabin"].astype("category").cat.codes

trainDf.drop("Cabin", axis=1, inplace=True)

trainDf.head()
cols = trainDf.columns.tolist()

cols.remove('Survived')

print(cols)

survived=trainDf[trainDf.Survived == 1]

survived.hist(figsize=(10,5), layout=(2,5))

died=trainDf[trainDf.Survived == 0]

died.hist(figsize=(10, 5), layout=(2,5))
trainDf.drop("CabinNo", axis=1, inplace=True)
def split_training_test_set(trainDf):

    # calculate number of samples in training set

    m = len(trainDf.index)

    # create array of indices and shuffle those

    Idx = np.arange(0, m)

    np.random.shuffle(Idx) 

    # select the first ~60% for the training data, and the rest for the cross-validation data

    m_train = math.ceil(m * 0.6)

    m_cv = m - m_train

    #print(m, m_train, m_cv)

    training = trainDf.iloc[Idx[0:m_train]]

    cv_set = trainDf.iloc[Idx[m_train:]]

    

    return [training, cv_set]
[training, cv_set] = split_training_test_set(trainDf)
training1 = training.drop(['Age'], axis=1)

print(training1.columns)

cv_set1 = cv_set.drop(['Age'], axis=1)

print(cv_set1.columns)
from sklearn.linear_model import LogisticRegression

LogReg1 = LogisticRegression(penalty='l2')

## 'Age' contains missing values, skip for now and see how good we get without it.

LogReg1.fit(training1.drop(['Survived', 'PassengerId'], axis=1), training1.Survived)
## Use the built-in 'score' method to calculate the accuracy of this model on the training and the cross-validation set. 

## 'score' gives the same answer as calculating the percentage of correctly predicted samples out of the set.

train_score = LogReg1.score(training1.drop(['Survived', 'PassengerId'], axis=1), training1.Survived)

cv_score = LogReg1.score(cv_set1.drop(['Survived', 'PassengerId'], axis=1), cv_set1.Survived)
print("Correctness on the training samples: {:g}".format(train_score))

print("Correctly predicted CV samples: {:g}".format(cv_score))

print("Baseline prediction correctness (all females survived): {:g}".format(len(cv_set1[cv_set1.IsFemale == 1])/len(cv_set1)))
## Remove the rows with missing values

trainDfAge = trainDf.dropna(axis=0, inplace=False)

[trainingAge, cv_setAge] = split_training_test_set(trainDfAge)

LogRegAge = LogisticRegression(penalty='l2')

LogRegAge.fit(trainingAge.drop(['Survived', 'PassengerId'], axis=1), trainingAge.Survived)

## It looks like this gives a notion on how well the learned logistic regression predicts the training data

train_scoreAge = LogRegAge.score(trainingAge.drop(['Survived', 'PassengerId'], axis=1), trainingAge.Survived)

cv_scoreAge = LogRegAge.score(cv_setAge.drop(['Survived', 'PassengerId'], axis=1), cv_setAge.Survived)

all_female_score = len(cv_setAge[cv_setAge.IsFemale == 1])/len(cv_setAge)

print("Accuracy on training set: {:g}".format(train_scoreAge))

print("Accuracy on cv set: {:g}".format(cv_scoreAge))

print("Accuracy of baseline prediction: {:g}".format(all_female_score))
#testDf.drop("Cabin", axis=1, inplace=True)

testDf["IsFemale"] = testDf["Sex"].astype("category").cat.codes

testDf["EmbarkPort"] = testDf["Embarked"].astype("category").cat.codes

testDf.drop("Sex", axis=1, inplace=True)

testDf.drop("Embarked", axis=1, inplace=True)

testDf.drop(['Name', 'Ticket', 'Age', 'Cabin'], axis=1, inplace=True)

testDf.info()

testDf.head()
modeFare = testDf.Fare.mode()

#print(modeFare.iloc[0])

isNull = pd.isnull(testDf["Fare"])

testDf[isNull]

testDf["Fare"].fillna(modeFare.iloc[0], inplace=True)

testDf.info()

testDf.iloc[152]
testDf1 = testDf.drop('PassengerId', axis=1, inplace=False)

print(testDf1.columns)

Predict1 = LogReg1.predict(testDf1)
answer1 = pd.DataFrame(Predict1, columns=["Survived"])

answer1["PassengerId"] = 0

answer1.PassengerId = testDf.PassengerId

answer1.info()

answer1.head()
answer1[['PassengerId', 'Survived']].astype('int32').to_csv("LogReg_output.csv", index=False)
from sklearn import tree



DecTree = tree.DecisionTreeClassifier()

DecTreeModel = DecTree.fit(training1.drop(['Survived', 'PassengerId'], axis=1), training1.Survived)

cv_score = DecTreeModel.score(cv_set1.drop(['Survived', 'PassengerId'], axis=1), cv_set1.Survived)

print(cv_score)