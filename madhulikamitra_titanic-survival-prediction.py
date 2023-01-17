# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



train.head(3)
train.isnull().sum(axis=0)

train=train.drop(columns='Cabin')

train['Age'] = train.Age.interpolate(method='akima',limit=None)

# test['Age'] = test.Age.astype('int')

# display(test.info())

#print (test[test.Age.isnull()])

meanAge=train['Age'].mean()

train['Age']=train['Age'].fillna(meanAge)

train['Age'] = train.Age.astype('int')
sns.countplot(train['Embarked'])

train = train.fillna({"Embarked": "S"})

train.isnull().sum(axis=0)
test.isnull().sum(axis=0)

test=test.drop(columns='Cabin')

test.head(3)

test['Age'] = test.Age.interpolate(method='akima',limit=None)

# test['Age'] = test.Age.astype('int')

# display(test.info())

#print (test[test.Age.isnull()])

meanAge=test['Age'].mean()

test['Age']=test['Age'].fillna(meanAge)
test['Age'] = test.Age.astype('int')
train=pd.get_dummies(train, columns=['Sex','Embarked'])

test=pd.get_dummies(test, columns=['Sex','Embarked'])

#train.head(3)

display(train.info())
features=['Pclass','Sex_female','Sex_male','Age']

X=train[features]

y=train['Survived']

test_X=test[features]



LRModel=LogisticRegression(solver='liblinear', random_state=0)

LRModel.fit(X,y)



LRPreds = LRModel.predict(test_X)



print(cross_val_score(LRModel, test_X, LRPreds, cv=5, scoring='accuracy').mean())
features=['Pclass','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Age']

X=train[features]

y=train['Survived']

test_X=test[features]



LRModel=LogisticRegression(solver='liblinear', random_state=0)

LRModel.fit(X,y)



LRPreds = LRModel.predict(test_X)

print(cross_val_score(LRModel, test_X, LRPreds, cv=5, scoring='accuracy').mean())





submission = pd.DataFrame({

        "PassengerId": test['PassengerId'],

        

        "Survived": LRPreds

    })

submission.count

os.chdir(r'/kaggle/working')

submission.to_csv(r'submission.csv', index=False)

from IPython.display import FileLink

FileLink(r'submission.csv')