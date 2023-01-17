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
# first we load in the train dataset and take a peek at it...

train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
# ...and do the same for the test data...

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
train = train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

train.head()
train['Sex'] = train['Sex'].replace({'male':1, 'female':0})

train.head()
train['Embarked'].unique()
train = pd.get_dummies(train)

train.head()
train.isna().sum()
median_age = train['Age'].median()

median_age
train['Age'] = train['Age'].fillna(median_age)
y = train['Survived']

X = train.drop('Survived', axis=1)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model.fit(X_train, y_train);
preds = model.predict(X_valid)

preds[:10]
def accuracy(preds, target): return (preds==target).sum()/len(preds)

accuracy(preds, y_valid)
test = pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId = test['PassengerId'] # we save the passenger ids because we'll need them for our submission

test = test.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

test['Sex'] = test['Sex'].replace({'male':1, 'female':0})

test['Age'] = test['Age'].fillna(median_age)

test['Fare'] = test['Fare'].fillna(train['Fare'].mean()) # it turned out that test had a missing value in Fare that needed to be filled in

test = pd.get_dummies(test)

test.head()
preds = model.predict(test)

sub = pd.DataFrame({'PassengerId': PassengerId, 'Survived': preds})

sub.head()
sub.to_csv('survived.csv', index=False)