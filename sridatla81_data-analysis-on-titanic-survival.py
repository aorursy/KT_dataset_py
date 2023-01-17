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
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.head()
df_train.columns
df_train.info()
df_train['PassengerId'].unique()
df_train.Survived.unique()
df_train.Pclass.unique()
df_train.Sex.unique()
df_train.Age.unique()
df_train.SibSp.unique()
df_train.Parch.unique()
df_train.Ticket.unique()
df_train.Fare.unique()
df_train.Cabin.unique()
df_train.Embarked.unique()
df_train[df_train['Embarked'].isnull() ]
df_train.isnull().sum()
df_test.isnull().sum()
df_train.info()
df_train.describe()
df_train.drop(columns=['Cabin'], inplace=True)

df_test.drop(columns=['Cabin'], inplace=True)
df_train.isnull().sum()
df_train.Age.isnull().sum()/df_train.index.size * 100
df_train.drop(columns=['Age'], inplace = True)

df_test.drop(columns=['Age'], inplace = True)
df_train.isnull().sum()
df_test.isnull().sum()
df_train = df_train[~df_train.Embarked.isnull()]
df_train.isnull().sum()
df_test = df_test[~df_test.Fare.isnull()]
df_test.isnull().sum()
df_train = df_train.drop_duplicates(keep='first')
df_train.shape
df_train.head()
df_train.drop(columns=['Name', 'Ticket'], inplace=True)

df_test.drop(columns=['Name', 'Ticket'], inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns
sns.pairplot(df_train)

plt.show()
df_train.describe()
df_train['Sex'] = df_train['Sex'].map({'male':1, 'female':0})

df_test['Sex'] = df_test['Sex'].map({'male':1, 'female':0})
embark_train = pd.get_dummies(df_train['Embarked'], drop_first = True)

embark_test = pd.get_dummies(df_test['Embarked'], drop_first = True)
df_train = pd.concat([df_train, embark_train], axis=1)

df_test = pd.concat([df_test, embark_test], axis=1)
df_train.drop(columns=['PassengerId', 'Embarked'], inplace=True)

df_test.drop(columns=['Embarked'], inplace=True)
df_train.head()
X_train = df_train.drop(columns=['Survived'], axis=1)

Y_train = df_train['Survived']

X_test = df_test.drop(columns=['PassengerId'], axis=1).copy()
# machine learning

from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, Y_train)
Y_pred = logistic_reg.predict(X_test)
logistic_reg.score(X_train, Y_train)
titanic_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': Y_pred})
titanic_submission.to_csv('titanic.csv', index=False)