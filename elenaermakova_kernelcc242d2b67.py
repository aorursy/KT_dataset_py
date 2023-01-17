# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
answers = pd.read_csv('../input/gender_submission.csv')
train.describe(include='all')
train.corr()
print(train[['Sex', 'Survived']].groupby(['Sex']).mean())
print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())
train.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Age'], inplace = True)
test.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Age'], inplace = True)
for df in [train, test]:
    print(df.shape)
    print()
    print(df.isna().sum())
for df in [train]:
    df.dropna(subset = ['Embarked'], inplace = True)
print(test[test['Fare'].isnull()])
test['Fare'].fillna(test[test['Pclass'] == 3].Fare.median(), inplace = True)
[train, test] = [pd.get_dummies(data = df, columns = ['Sex', 'Embarked']) for df in [train, test]]
train.head()
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = train[['Pclass', 'Sex_female', 'Sex_male', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']], train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1)
classifier = SVC()
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_val)
print(accuracy_score(y_pred, y_val))
X_train, y_train = train[['Pclass', 'Sex_female', 'Sex_male', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']], train['Survived']
X_test, y_test = test[['Pclass', 'Sex_female', 'Sex_male', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']], answers['Survived']
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_pred, y_test))
test.head()
test_pId = test.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':test_pId, 'Survived':y_test})
print(my_submission['Survived'].value_counts())
my_submission.to_csv('submission.csv', index = False)