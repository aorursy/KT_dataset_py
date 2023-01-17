#load libraries 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("../titanic/train.csv")
test = pd.read_csv("../titanic/test.csv")
train.head()
test.head()
import os
os.path.realpath('.')
train.describe()
train.corr()
traindf = train[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Sex', 'Parch']]
pd.isna(train).sum()
newdf = traindf.fillna(traindf.mean())
pd.isna(newdf).sum()
newdf.head()
testdf = test[['Pclass', 'Age', 'Fare', 'SibSp', 'Sex','Parch']]
testdata = testdf.fillna(testdf.mean())
col = ['Sex']
newdf[col] = newdf[col].apply(lambda x: pd.factorize(x)[0] + 1)
col = ['Sex']
testdata[col] = testdata[col].apply(lambda x: pd.factorize(x)[0] + 1)
y = newdf['Survived']
x = newdf[['Pclass', 'Age', 'Fare', 'SibSp','Sex', 'Parch']]
model = LogisticRegression()
model.fit(x, y)
Survived = model.predict(testdata)
PassengerId = test['PassengerId']
Submission = pd.DataFrame(columns=[PassengerId,Survived])
Submission
Submission = Submission.T
Submission.to_csv('Submission.csv')
clf = DecisionTreeClassifier()
clf.fit(x, y)
Survived1 = clf.predict(testdata)
Survived1
Submission2 = pd.DataFrame(columns=[PassengerId,Survived1])
Submission2 = Submission2.T
Submission2.to_csv('Submission2.csv')


