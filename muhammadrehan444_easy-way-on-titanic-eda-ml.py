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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
a=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
test.head()
train.isnull().sum().sort_values(ascending=False)
train.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

train['Age']= train['Age'].fillna(train['Age'].mean())

train['Embarked'].value_counts()

train.dropna(subset=['Embarked'],axis = 0 ,inplace = True)
train = pd.get_dummies(data=train, columns=['Embarked'])
train = pd.get_dummies(data=train, columns=['Sex'])
train['Fare'] = train['Fare'].astype(int, copy=True)
train['Age'] = train['Age'].astype(int, copy=True)
import matplotlib.pyplot as plt 
plt.hist(train['Age'])
plt.show()
bins = np.linspace(min(train['Age']), max(train['Age']), 9 )
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
train['New_age']= pd.cut(train['Age'], bins, labels=label, include_lowest=True)
train = pd.get_dummies(data=train, columns=['New_age'])
train.drop(['Age'], axis=1, inplace=True)

import matplotlib.pyplot as plt 
plt.hist(train['Fare'])
plt.show()
bins = np.linspace(min(train['Fare']), max(train['Fare']),  11)
label = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK']
train['New_Fare']= pd.cut(train['Fare'], bins, labels=label, include_lowest=True)
train = pd.get_dummies(data=train, columns=['New_Fare'])
train.drop(['Fare'], axis=1, inplace=True)
train.set_index('PassengerId',inplace = True)
train.head()
test.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
test['Age']= test['Age'].fillna(test['Age'].mean())
test.dropna(subset=['Embarked'],axis = 0 ,inplace = True)
test = pd.get_dummies(data=test, columns=['Embarked'])
test = pd.get_dummies(data=test, columns=['Sex'])
test['Fare']= test['Fare'].fillna(test['Fare'].mean())
test['Fare'] = test['Fare'].astype(int, copy=True)
test['Age'] = test['Age'].astype(int, copy=True)
bins = np.linspace(min(test['Age']), max(test['Age']), 9 )
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
test['New_age']= pd.cut(test['Age'], bins, labels=label, include_lowest=True)
test = pd.get_dummies(data=test, columns=['New_age'])
test.drop(['Age'], axis=1, inplace=True)
bins = np.linspace(min(test['Fare']), max(test['Fare']),  11)
label = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK']
test['New_Fare']= pd.cut(test['Fare'], bins, labels=label, include_lowest=True)
test = pd.get_dummies(data=test, columns=['New_Fare'])
test.drop(['Fare'], axis=1, inplace=True)
test.set_index('PassengerId',inplace = True)
test.head()
X_train = train.drop(['Survived'], axis=1)
Y_train= train['Survived']

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
