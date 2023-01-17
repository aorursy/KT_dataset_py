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
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
test.info()
train.isnull().sum()
# AGE

train['Age'] = train['Age'].fillna(value=train['Age'].median())

test['Age'] = test['Age'].fillna(value=test['Age'].median())
test.isnull().sum()
# FARE

train['Fare'] = train['Fare'].fillna(value=train['Fare'].median())

test['Fare'] = test['Fare'].fillna(value=test['Fare'].median())
train['Embarked'].unique()
# Embarked

train['Embarked'] = train['Embarked'].fillna(value=train['Embarked'].mode()[0])

test['Embarked'] = test['Embarked'].fillna(value=test['Embarked'].mode()[0])
train['Cabin'].fillna(value='Missing',inplace=True)
train['Cabin'].isnull().sum()
# train Cabin

train['Cabin'].fillna(value='Missing',inplace=True)

train['Cabin']=train['Cabin'].apply(lambda x : x[0])
train['Cabin'].value_counts()
# test Cabin

test['Cabin'].fillna(value='Missing',inplace=True)

test['Cabin']=test['Cabin'].apply(lambda x : x[0])
# extract title from name

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#We will combine a few categories, since few of them are unique for train data set

train['Title'] = train['Title'].replace(['Capt', 'Dr', 'Major', 'Rev'], 'Officer')

train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Royal')

train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')

train['Title'] = train['Title'].replace(['Mme'], 'Mrs')

train['Title'].value_counts()

#We will combine a few categories, since few of them are unique for test data set

test['Title'] = test['Title'].replace(['Capt', 'Dr', 'Major', 'Rev'], 'Officer')

test['Title'] = test['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Royal')

test['Title'] = test['Title'].replace(['Mlle', 'Ms'], 'Miss')

test['Title'] = test['Title'].replace(['Mme'], 'Mrs')

test['Title'].value_counts()

#Family Size & Alone train

train['Family_Size'] = train['SibSp'] + train['Parch'] + 1

train['IsAlone'] = 0

train.loc[train['Family_Size']==1, 'IsAlone'] = 1

#Family Size & Alone test

test['Family_Size'] = test['SibSp'] + test['Parch'] + 1

test['IsAlone'] = 0

test.loc[test['Family_Size']==1, 'IsAlone'] = 1

all = pd.concat([train, test], sort = False)

all.info()

all_dummies = pd.get_dummies(all.drop(['Name','Ticket'],axis=1), drop_first = True)

all_dummies.head()

all_train = all_dummies[all_dummies['Survived'].notna()]

all_train.info()

all_test = all_dummies[all_dummies['Survived'].isna()]

all_test.info()

y=all_train['Survived']

X=all_train.drop(['Survived','PassengerId'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression
logModel=LogisticRegression(max_iter=5000)

logModel.fit(X_train,y_train)
prediction=logModel.predict(X_test)

prediction
from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))
logModel.score(X_train,y_train)
logModel.score(X_test,y_test)
X_Submission = all_test.drop(['PassengerId', 'Survived'], axis = 1)

pred_for_submission = logModel.predict(X_Submission).astype(int)

logSub = pd.DataFrame({'PassengerId': all_test['PassengerId'], 'Survived':pred_for_submission })

logSub.head()

logSub.to_csv('1_Logistic_Regression_Submission.csv',index=False)