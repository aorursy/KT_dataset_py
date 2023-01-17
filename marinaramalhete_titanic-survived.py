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
from sklearn.tree import DecisionTreeClassifier

#from sklearn.ensemble import RandomForestClassifier
gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')
gender.head()
test.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

train.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
test.head()
train.head()
new_train = pd.get_dummies(train)

new_test = pd.get_dummies(test)
new_train.head()
new_test.head()
new_test['Age'].fillna(new_test['Age'].mean(), inplace = True)

new_test['Fare'].fillna(new_test['Fare'].mean(), inplace = True)
new_train['Age'].fillna(new_train['Age'].mean(), inplace = True)
new_test.isnull().sum().sort_values(ascending = False).head(10)
new_train.isnull().sum().sort_values(ascending = False).head(10)
# separando features (input) e target para criacao do modelo

X =  new_train.drop('Survived', axis = 1)

y = new_train['Survived']
# modelo decision tree

tree = DecisionTreeClassifier(max_depth = 3, random_state = 0)

tree.fit(X, y)

tree.score(X, y)
# # modelo Random Forest

# rf = RandomForestClassifier(n_estimators = 3, random_state = 0)

# rf.fit(X, y)

# rf.score(X, y)
# submission = pd.DataFrame()

# submission['PassengerId'] = new_test['PassengerId']

# submission['Survived'] = rf.predict(new_test)
submission = pd.DataFrame()

submission['PassengerId'] = new_test['PassengerId']

submission['Survived'] = tree.predict(new_test)
submission.to_csv('submission.csv', index = False)