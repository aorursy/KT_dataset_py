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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
train.tail()
test.tail()
train.info()
test.info()
train.describe()
test.describe()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.groupby('Survived').size()
import matplotlib.pyplot as plt

import seaborn as sns

g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cmap='plasma')
sns.heatmap(test.isnull(), yticklabels = False, cmap = 'plasma')
train.isnull().sum()
train['Age'] = train.fillna(train['Age'].mean())
train.isnull().sum()
train = train.drop(['Cabin'], axis = 1)
train.isnull().sum()
test.isnull().sum()
test['Age'] = test.fillna(test['Age'].mean())
test = test.drop(['Cabin'], axis = 1)
test[test['Fare'].isnull()]
test.set_value(152, 'Fare', 50)
test.isnull().sum()
sex_train = pd.get_dummies(train['Sex'],drop_first=True)

embark_train = pd.get_dummies(train['Embarked'], drop_first =True)
sex_test = pd.get_dummies(test['Sex'], drop_first = True)

embark_test = pd.get_dummies(test['Embarked'], drop_first =True)
train.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)
test.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)
train_f = pd.concat([train, sex_train, embark_train], axis=1)

train_f.drop(['PassengerId'], axis = 1, inplace = True)

train_f.head()
test_f = pd.concat([test, sex_test, embark_test], axis = 1)

test_f.drop(['PassengerId'], axis = 1, inplace = True)

test_f.head()
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm , tree

import xgboost

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import metrics



features = ['Pclass','Age', 'male']

target = 'Survived'

X = train_f[features]

y = train_f[target]

X_test_f = test_f[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
clf = RandomForestClassifier(n_estimators=100, max_depth = 5,min_samples_split=4, min_samples_leaf=5,)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy using Random Forest:",metrics.accuracy_score(y_test, y_pred))

print(len(y_pred))
y_pred_f = clf.predict(X_test_f)

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred_f})

print(submission)
import os

filename = "titanic_kaggle.csv"

submission.to_csv(filename , index = False)

print('saved file ' + filename)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy using Logistic Regression:', metrics.accuracy_score(y_test, y_pred))
svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print("Accuracy using SVC:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))