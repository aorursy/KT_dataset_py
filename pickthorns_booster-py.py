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
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
import matplotlib as plt
import xgboost as xgb
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 600)
print(test.head())
test.fillna(0)
train.fillna(0)
train['Survived']
train = train.drop(['Name'], axis=1)
train['Embarked'] = train['Embarked'].replace(['S','Q','C'],[0,1,2])
print(train.head())
train['Sex'] = train['Sex'].replace(['male','female'],[0,1])
train = train.drop('Cabin', axis=1)

train = train.drop('Ticket', axis=1)
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
test = test.drop(['Name', 'Cabin', 'Ticket'], axis=1)
test['Sex'] = test['Sex'].replace(['male','female'], [0,1])
test['Embarked'] = test['Embarked'].replace(['Q','S','C'], [0,1,2])
test['Age'] = test['Age'].fillna(np.mean(test['Age']))
test
target = train['Survived']
train = train.drop('Survived', axis=1)
xgtrain = xgb.DMatrix(train.values, target.values)
xgtest = xgb.DMatrix(test.values)

bst = xgb.XGBClassifier().fit(train.values, target.values)

predictions = bst.predict(test.values)
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)