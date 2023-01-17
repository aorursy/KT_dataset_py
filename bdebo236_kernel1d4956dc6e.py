# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
titanictrain = pd.read_csv("/kaggle/input/titanic/train.csv")

titanictest = pd.read_csv("/kaggle/input/titanic/test.csv")
titanictrain.head()
titanictrain.info()
grouped = titanictrain.groupby(['Sex','Pclass'])
grouped_test = titanictest.groupby(['Sex','Pclass'])
print(grouped.Age.median())

print(grouped_test.Age.median())
titanictrain.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

titanictest.Age = grouped_test.Age.apply(lambda x: x.fillna(x.median()))
titanictrain.Cabin = titanictrain.Cabin.fillna('U')

titanictest.Cabin = titanictest.Cabin.fillna('U')
most_embarked = titanictrain.Embarked.value_counts().index[0]

titanictrain.Embarked = titanictrain.Embarked.fillna(most_embarked)

titanictrain.Fare = titanictrain.Fare.fillna(titanictrain.Fare.median())



most_embarked_test = titanictest.Embarked.value_counts().index[0]

titanictest.Embarked = titanictest.Embarked.fillna(most_embarked_test)

titanictest.Fare = titanictest.Fare.fillna(titanictest.Fare.median())
titanictrain.info()
titanictest.info()
titanictrain.Cabin = titanictrain.Cabin.map(lambda x: x[0])

titanictest.Cabin = titanictest.Cabin.map(lambda x: x[0])
titanictrain.Sex = titanictrain.Sex.map({"male": 0, "female":1})



pclass_train_dummies = pd.get_dummies(titanictrain.Pclass, prefix="Pclass")

cabin_train_dummies = pd.get_dummies(titanictrain.Cabin, prefix="Cabin")

embarked_train_dummies = pd.get_dummies(titanictrain.Embarked, prefix="Embarked")



titanic_train_dummies = pd.concat([titanictrain, pclass_train_dummies, cabin_train_dummies, embarked_train_dummies], axis=1)



titanic_train_dummies.drop(['Pclass', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



titanic_train_dummies.head()





titanictest.Sex = titanictest.Sex.map({"male": 0, "female":1})



pclass_test_dummies = pd.get_dummies(titanictest.Pclass, prefix="Pclass")

cabin_test_dummies = pd.get_dummies(titanictest.Cabin, prefix="Cabin")

embarked_test_dummies = pd.get_dummies(titanictest.Embarked, prefix="Embarked")



titanic_test_dummies = pd.concat([titanictest, pclass_test_dummies, cabin_test_dummies, embarked_test_dummies], axis=1)



titanic_test_dummies.drop(['Pclass', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



titanic_test_dummies.head()
passengerId = titanic_train_dummies.PassengerId

print(passengerId)

titanic_train_dummies.drop(['PassengerId'], axis = 1, inplace = True)



passengerId_test = titanic_test_dummies.PassengerId

print(passengerId_test)

titanic_test_dummies.drop(['PassengerId'], axis = 1, inplace = True)
titanic_train_dummies.head()
titanic_test_dummies.head()
result = titanic_train_dummies.Survived

titanic_train_dummies.drop(['Survived'], axis = 1, inplace = True)

titanic_train_dummies.head()
titanic_test_dummies.head()
titanic_test_dummies['Cabin_T'] = 0
model = RandomForestClassifier()

model.fit(titanic_train_dummies, result)
result_test = model.predict(titanic_test_dummies)
titanic_result = pd.DataFrame({'PassengerId': list(passengerId_test), 'Survived': result_test})
titanic_result.to_csv('/kaggle/working/titanic_pred.csv', index=False)