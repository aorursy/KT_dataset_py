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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
##Replace the missing values with the mean in the age

train['Age'] = train['Age'].fillna(train['Age'].mean()).astype(int)

test['Age'] = test['Age'].fillna(test['Age'].mean()).astype(int)

train['Fare'] = train['Fare'].fillna(train['Fare'].median())

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

##data processing

fulltxt = [train, test]

train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')

for txt in fulltxt:

    txt['Sex'] = txt['Sex'].map({'female':1, 'male': 0}).astype(int)

    txt['Embarked'] = txt['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    txt['Cabin']= txt['Cabin'].apply(lambda x : 0 if type(x) == float else 1)

    txt['Age'] = txt['Age'].astype(int)

    txt.loc[txt['Age'] <= 16, 'Age'] = 0

    txt.loc[(txt['Age'] > 16) & (txt['Age'] <= 32),'Age'] = 1

    txt.loc[(txt['Age'] > 32) & (txt['Age'] <= 64), 'Age'] = 2

    txt.loc[txt['Age'] > 64, 'Age'] = 3

    txt['FamilySize'] = txt['SibSp'] + txt['Parch']

    txt.loc[txt['Fare'] <= 7.91,'Fare'] = 0

    txt.loc[(txt['Fare'] > 7.91) & (txt['Fare'] <= 14.454), 'Fare'] = 1

    txt.loc[(txt['Fare'] > 14.454) & (txt['Fare'] <= 31), 'Fare'] = 2

    txt.loc[txt['Fare'] > 31, 'Fare'] = 3

    txt['Fare'] = txt['Fare'].astype(int)

    #txt.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis = 1)
#data processing

train = train.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis = 1)

test = test.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis = 1)

# train = train.drop(['PassengerId'], axis = 1)

# test = test.drop(['PassengerId'], axis = 1)
#train.head()

##data split

train_x = train.copy().drop(['PassengerId'],axis = 1).ix[:,1:]

train_x.head()

train_y = train.copy().ix[:,1]

test_x = test.copy().ix[:,1:]
test.head()
#data split

from xgboost import XGBClassifier # import xgboost

model = XGBClassifier() #import model

model.fit(train_x, train_y) #train model

y_pred = model.predict(test_x) #model predict

print(y_pred)
#predict accuracy

from sklearn.metrics import classification_report

test_y = pd.read_csv('../input/gender_submission.csv')

# test_y.Survived

# print(classification_report(y_pred, test_y.Survived))

print(model.score(test_x, test_y[['Survived']]))
StackingSubmission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': y_pred })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)