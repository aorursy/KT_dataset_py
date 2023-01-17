import matplotlib.pyplot as plt

%matplotlib inline

import random

import numpy as np

import pandas as pd

from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics

import sklearn.ensemble as ske
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train
test
# count of male and female

bar1 = train.groupby(['Sex'])['Sex'].count()



bar1.plot.bar()
# average fare by class

bar2 = train.groupby(['Pclass'])['Fare'].mean()



bar2.plot.bar()
# Number of passengers from each point of departure

bar3 = train.groupby(['Embarked'])['Embarked'].count()



bar3.plot.bar()
# Average Age by Passenger Class



bar4 = train.groupby(['Pclass'])['Age'].mean()



bar4.plot.bar()
# Survival rates by age bins

age_bins = pd.cut(train["Age"], np.arange(0, 90, 10))

bar5 = train.groupby(age_bins).mean()

bar5['Survived'].plot.bar()
# Survival rates by class

bar6 = train.groupby(["Pclass"]).mean()

bar6['Survived'].plot.bar()
# drop those variables from both train and test sets

train = train.drop(['PassengerId','Name','Cabin','Ticket','Parch','SibSp','Embarked'], axis=1)

test = test.drop(['Name','Cabin','Ticket','Parch','SibSp','Embarked'], axis=1)
train.count()
# find the averages for age . . .

train["Age"].mean(skipna=True)
# . . . and fare

train["Fare"].mean(skipna=True)
train["Age"].fillna(29.7, inplace=True)

train["Fare"].fillna(34.69, inplace=True)



test["Age"].fillna(29.7, inplace=True)

test["Fare"].fillna(34.69, inplace=True)
train.count()
train.dtypes
train['Sex'].replace(['female','male'],[0,1],inplace=True)

test['Sex'].replace(['female','male'],[0,1],inplace=True)
train['Sex']
X = train.drop(["Survived"], axis = 1).values

Y = train["Survived"].values
train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size = 0.2)
model = ske.RandomForestClassifier(n_estimators=100)
model.fit(train_X , train_Y)
print (model.score( train_X , train_Y ) , model.score( test_X , test_Y ))
submission_data = test.loc[:, test.columns != 'PassengerId']

submission_predictions = model.predict(submission_data)
my_submission = pd.DataFrame({'PassengerId': test["PassengerId"], 'Survived': submission_predictions})



my_submission.to_csv('submission.csv', index=False)