# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
import matplotlib.pyplot as plt

import seaborn as sns

graph = sns.countplot(train_data['Survived'])
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
print(train_data.columns)

train_data.head(5)
train_data = train_data.drop(['Cabin', 'Ticket','Name'],axis = 1)

train_data = train_data.drop(['Fare'],axis = 1)
train_data.head()
test_data = test_data.drop(['Cabin', 'Ticket', 'Name'], axis = 1)

test_data = test_data.drop(['Fare'],axis = 1)

test_data.head()
print(pd.isnull(train_data).sum())
train_data = train_data.fillna({"Embarked": "S"})

test_data = test_data.fillna({"Embarked": "S"})

train_data = train_data.fillna({"Age": "young"})

print(pd.isnull(train_data).sum())

train_data = train_data.drop(['Age'],axis = 1)

test_data = test_data.drop(['Age'],axis = 1)
train_data.Embarked.replace(to_replace = dict(S=1,Q=2,C=3), inplace=True)

test_data.Embarked.replace(to_replace = dict(S=1,Q=2,C=3), inplace=True)

train_data.Sex.replace(to_replace = dict(male=0,female=1), inplace=True)

test_data.Sex.replace(to_replace = dict(male=0,female=1), inplace=True)
train_data.head()

test_data.head()
prediction_class = train_data.drop(['PassengerId', 'Survived'],axis = 1) 

target = train_data['Survived']

x_train, x_test, y_train, y_test = train_test_split(prediction_class, target, test_size=0.33, random_state=0)
reg = LogisticRegression()

model = reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

print(y_pred)

accuracy = accuracy_score(y_test, y_pred) * 100

print(accuracy)
ids = test_data['PassengerId']

predictions = reg.predict(test_data.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('output.csv', index=False)

output_data = pd.read_csv('output.csv')

print(output_data)