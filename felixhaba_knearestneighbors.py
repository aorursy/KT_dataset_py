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

test = pd.read_csv("/kaggle/input/titanic/test.csv")

X_test = test

train
X_test
train.columns
X_test
y = train.Survived

X = train.drop(["Survived", "Ticket", "Name"], axis=1)

X_test = test.drop(["Ticket", "Name"], axis=1)
X.Cabin.fillna("Missing", inplace=True)

X_test.Cabin.fillna("Missing", inplace=True)
X
X_test
X.isna().sum()
X_test.isna().sum()
X_test.Fare.fillna(X_test.Fare.mean(), inplace=True)

X.Embarked.fillna(X.Embarked.mode()[0], inplace=True)

X_test.Age.fillna(X.Age.mean(), inplace=True)

X.Age.fillna(X.Age.mean(), inplace=True)
X_test
X.Embarked.mode()[0]
X.drop("Cabin", axis=1, inplace=True)

X_test.drop("Cabin", axis=1, inplace=True)
X.set_index(X.PassengerId, drop = True)
X_test
X_test = X_test.set_index(X_test.PassengerId, drop=True).drop(["PassengerId"], axis=1)

X = X.set_index(X.PassengerId, drop=True).drop(["PassengerId"], axis=1)
X_test
X.Sex.nunique()
X_test
X = X.merge(pd.get_dummies(X.Embarked), left_index=True, right_index=True)

X = X.merge(pd.get_dummies(X.Sex), left_index=True, right_index=True)

X.drop(["Sex", "Embarked"], axis=1, inplace=True)

X_test = X_test.merge(pd.get_dummies(X_test.Embarked), left_index=True, right_index=True)

X_test = X_test.merge(pd.get_dummies(X_test.Sex), left_index=True, right_index=True)

X_test.drop(["Sex", "Embarked"], axis=1, inplace=True)

train = train.merge(pd.get_dummies(train.Embarked), left_index=True, right_index=True)

train = train.merge(pd.get_dummies(train.Sex), left_index=True, right_index=True)

train.drop(["Sex", "Embarked"], axis=1, inplace=True)
X
X_test
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

X_train, X_valid, y_train, y_valid = train_test_split(X,y)
knn.fit(X_train, y_train)
help(knn.score)
knn.score(X_valid, y_valid)
knn.fit(X, y)
X_test
predictions = knn.predict(X_test)

result = pd.DataFrame({"Survived": predictions}, index=X_test.index)

submission = result.to_csv("submission.csv")

result
train
train.corr()
preds = knn.predict(X)

(preds == y.values).mean()