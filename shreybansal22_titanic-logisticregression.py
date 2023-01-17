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
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.shape
test.shape
train.dtypes
train.describe().T
train = train.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
train.isnull().sum()
train['Age'] = train['Age'].fillna(train['Age'].median())

train['Embarked'] = train['Embarked'].fillna('S')
train.isnull().sum()
train.info()
train = train.drop(['Name'], axis = 1)
train.info()
cat_columns=train.drop(train.select_dtypes(exclude=['object']), axis=1).columns
print(cat_columns)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
train[cat_columns[0]] = labelencoder.fit_transform(train[cat_columns[0]].astype('str'))
labelencoder1=LabelEncoder()
train[cat_columns[1]] = labelencoder.fit_transform(train[cat_columns[1]].astype('str'))
train.info()
train.head()
import seaborn as sns
import matplotlib.pyplot as plt
sns.FacetGrid(train, col='Survived').map(plt.hist, 'Embarked')
sns.FacetGrid(train, col='Survived').map(plt.hist, 'Age')
sns.FacetGrid(train, col='Survived').map(plt.hist, 'Sex')
sns.FacetGrid(train, col='Survived').map(plt.hist, 'Pclass')
sns.FacetGrid(train, col='Survived').map(plt.hist, 'Parch')
sns.FacetGrid(train, col='Survived').map(plt.hist, 'SibSp')
sns.FacetGrid(train, col='Survived').map(plt.hist, 'Fare')
train = train.drop(['Fare'], axis = 1)
X = train.drop(['Survived'], axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

regressor=LogisticRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
plt.scatter([i for i in range(len(X_test["Sex"]))], y_test, color='green')
plt.plot([i for i in range(len(X_test["Sex"]))], y_pred, color='blue')

plt.ylabel('no of Survived')
plt.xlabel('no of Passengers')

plt.show()
test.head()
passenger_id=test['PassengerId']
test = test.drop(['PassengerId', 'Ticket', 'Cabin','Name','Fare'], axis = 1)
test.dtypes
test['Age'] = test['Age'].fillna(train['Age'].median())

test['Embarked'] = test['Embarked'].fillna('S')
test.info()
cat_columns1=test.drop(test.select_dtypes(exclude=['object']), axis=1).columns
print(cat_columns1)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
test[cat_columns1[0]] = labelencoder.fit_transform(test[cat_columns1[0]].astype('str'))
labelencoder1=LabelEncoder()
test[cat_columns1[1]] = labelencoder.fit_transform(test[cat_columns1[1]].astype('str'))
test.head()
y_test_pred = regressor.predict(test)
plt.scatter([i for i in range(len(test['Sex']))], y_test_pred, color='blue')

plt.ylabel('no.of.Survived')
plt.xlabel('no.of.Passenger')

plt.show()
result = pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": y_test_pred
    })

result.to_csv('./submission.csv', index=False)
