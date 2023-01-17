import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('mode.chained_assignment', None)

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import LabelEncoder

import pylab as pl
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train.pivot_table('PassengerId', 'Sex', 'Survived', 'count').plot(kind='bar', stacked=True)
train.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
fig, axes = plt.subplots(ncols=2)

train.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0], title='SibSp')

train.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')
train.isnull().count()
train.PassengerId[train.Cabin.notnull()].count()
train.PassengerId[train.Age.notnull()].count()
train.Age = train.Age.median()
train[train.Embarked.isnull()]
MaxPassEmbarked = train.groupby('Embarked').count()['PassengerId']

train.Embarked[train.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
train.PassengerId[train.Fare.isnull()]
train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

train
label = LabelEncoder()

dicts = {}



label.fit(train.Sex.drop_duplicates()) 

dicts['Sex'] = list(label.classes_)

train.Sex = label.transform(train.Sex)



label.fit(train.Embarked.drop_duplicates())

dicts['Embarked'] = list(label.classes_)

train.Embarked = label.transform(train.Embarked)
train
test.Age[test.Age.isnull()] = test.Age.mean()

test.Fare[test.Fare.isnull()] = test.Fare.median()

MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']

test.Embarked[test.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

result = pd.DataFrame(test.PassengerId)

test = test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)



label.fit(dicts['Sex'])

test.Sex = label.transform(test.Sex)



label.fit(dicts['Embarked'])

test.Embarked = label.transform(test.Embarked)
test
target = train.Survived

predictors = train.drop(['Survived'], axis=1)

kfold = 5

itog_val = {} 
ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train, target, test_size=0.25)
model_rfc = RandomForestClassifier(n_estimators = 80, max_features='auto', criterion='entropy',max_depth=4)
model_rfc.fit(predictors, target)

result.insert(1,'Survived', model_rfc.predict(test))

result.to_csv('HuzaifahSaleem_predictions.csv', index=False)