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
files_dir = '/kaggle/input/titanic/'

train = pd.read_csv(files_dir+'train.csv')

train
train.isna().sum()
train[train.Embarked.isna()]
train.Embarked = train.Embarked.fillna('C')
y = train.Survived

X = train.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin', 'Survived'], axis=1)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
X = pd.get_dummies(X, columns=['Embarked', 'Sex'])

X.head()
X['family_size'] = X['SibSp'] + X['Parch']

X.head()
X.Fare.describe()
X['Fare_cat']=0

X.loc[X['Fare']<=7.91,'Fare_cat']=0

X.loc[(X['Fare']>7.91)&(X['Fare']<=14.454),'Fare_cat']=1

X.loc[(X['Fare']>14.454)&(X['Fare']<=31),'Fare_cat']=2

X.loc[(X['Fare']>31)&(X['Fare']<=513),'Fare_cat']=3

X.head()
dt = DecisionTreeClassifier()

dt.fit(X, y)
test = pd.read_csv(files_dir+'test.csv')

test.head()
x_test = test.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin'], axis=1)

x_test = pd.get_dummies(x_test, columns=['Embarked', 'Sex'])

x_test['family_size'] = x_test['SibSp'] + x_test['Parch']

x_test.head()
x_test['Fare_cat']=0

x_test.loc[x_test['Fare']<=7.91,'Fare_cat']=0

x_test.loc[(x_test['Fare']>7.91)&(x_test['Fare']<=14.454),'Fare_cat']=1

x_test.loc[(x_test['Fare']>14.454)&(x_test['Fare']<=31),'Fare_cat']=2

x_test.loc[(x_test['Fare']>31)&(x_test['Fare']<=513),'Fare_cat']=3

x_test.head()
x_test.isna().sum()
x_test[x_test.Fare.isna()]
x_test.groupby(['Sex_male', 'Pclass', 'Embarked_S']).Fare.mean()
x_test.Fare = x_test.Fare.fillna(12.718872)
y_pred = dt.predict(x_test)
subm = pd.read_csv(files_dir+'gender_submission.csv')

subm.head()
test_subm = test.copy()

test_subm['Survived'] = y_pred

test_subm = test_subm[['PassengerId', 'Survived']]

test_subm.head()
accuracy_score(y_pred, subm.Survived)
test_subm.to_csv('/kaggle/working/my_submissions_v1.csv', index=False)