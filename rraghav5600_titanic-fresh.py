# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter



from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



train.head()
for i in train.columns:

    tt = Counter(train[i].isnull().values)

    print(tt, i)

print('*'*30)

for i in test.columns:

    tt = Counter(test[i].isnull().values)

    print(tt, i)
train['Embarked'].value_counts()
train['Embarked'].replace(np.NaN, 'S', inplace=True)
train['FamilySize']=train['SibSp']+train['Parch']

test['FamilySize']=test['SibSp']+test['Parch']

for_submission = test['PassengerId']



train=train.drop(['Cabin','PassengerId','Name', 'Ticket', 'SibSp', 'Parch'],axis=1)

test=test.drop(['Cabin','PassengerId','Name', 'Ticket', 'SibSp', 'Parch'],axis=1)
for i in train.columns:

    tt = Counter(train[i].isnull().values)

    print(tt, i)

print('*'*30)

for i in test.columns:

    tt = Counter(test[i].isnull().values)

    print(tt, i)
train['Age'].fillna(train['Age'].mean(), inplace=True)

test['Age'].fillna(test['Age'].mean(), inplace=True)

test['Fare'].fillna(test['Fare'].mean(), inplace=True)
my_label = LabelEncoder()

train['Sex'] = my_label.fit_transform(train['Sex'])

test['Sex'] = my_label.fit_transform(test['Sex'])

my_label = LabelEncoder()

train['Embarked'] = my_label.fit_transform(train['Embarked'])

test['Embarked'] = my_label.fit_transform(test['Embarked'])

train
bucket=20

train['Age'] = [int(i) for i in ((train['Age'].values)//bucket)]

test['Age'] = [int(i) for i in ((test['Age'].values)//bucket)]

train['Fare'] = [int(i) for i in ((train['Fare'].values)//bucket)]

test['Fare'] = [int(i) for i in ((test['Fare'].values)//bucket)]



print(train.head())

print(test.head())
X = train.drop(['Survived'], axis=1)

y = train['Survived']
train_X, val_X, train_y, val_y = train_test_split(X, y)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1)

rf.fit(train_X, train_y)

prediction = rf.predict(val_X)

print(mean_absolute_error(val_y, prediction), accuracy_score(val_y, prediction))
from xgboost import XGBClassifier



XGB = XGBClassifier()

XGB.fit(train_X, train_y)

prediction = XGB.predict(val_X)

print(mean_absolute_error(val_y, prediction), accuracy_score(val_y, prediction))
final_rf = RandomForestClassifier(random_state=1)

final_xgb = XGBClassifier()



final_rf.fit(X,y)

final_xgb.fit(X,y)
vals = final_rf.predict(test)

file = pd.DataFrame({'PassengerId':for_submission, 'Survived':vals})

file.to_csv('submission_rf.csv', index = False)

file.head()
vals = final_xgb.predict(test)

file = pd.DataFrame({'PassengerId':for_submission, 'Survived':vals})

file.to_csv('submission_xgb.csv', index = False)

file.head()