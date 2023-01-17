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
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.isnull().sum()
train_data = train_data.drop(['Cabin'],1)

train_data = train_data.dropna()
train_data.isnull().sum()
y = train_data['Survived']

X = train_data.drop(['Survived','PassengerId','Name','Ticket'],1)

X['Sex'].replace(['female','male'],[0,1],inplace=True)
X = pd.get_dummies(X,['Embarked'])
X.head()
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()

model.fit(X,y)
test_data.shape
test_data = test_data.drop(['Cabin'],1)

test_data = test_data.fillna(0)

ids = test_data[['PassengerId']]

X_test = test_data.drop(['PassengerId','Name','Ticket'],1)

X_test['Sex'].replace(['female','male'],[0,1],inplace=True)

X_test = pd.get_dummies(X_test,['Embarked'])
X_test.shape
prediction = model.predict(X_test)
results = ids.assign(Survived = prediction) 

results.to_csv("titanic-results.csv", index=False)