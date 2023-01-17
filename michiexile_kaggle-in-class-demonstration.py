# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import scipy as sp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

titanic = pd.read_csv("../input/train.csv")

titanic
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
ohe = OneHotEncoder()

titanic = pd.read_csv("../input/train.csv")

titanic["CabinGroup"] = titanic["Cabin"].fillna("").str[:1]

y_train = titanic["Survived"].values
X_numeric = titanic[["Age", "Fare"]].fillna(0).values
X_categorical = ohe.fit_transform(titanic[["Pclass", "Sex", "Embarked", "CabinGroup"]].fillna("").values)
X_train = sp.hstack([X_numeric,X_categorical.todense()])

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)
titanic = pd.read_csv("../input/test.csv")
titanic["CabinGroup"] = titanic["Cabin"].fillna("").str[:1]
X_numeric = titanic[["Age", "Fare"]].fillna(0).values
X_categorical = ohe.transform(titanic[["Pclass", "Sex", "Embarked", "CabinGroup"]].fillna("").values)
X_test = sp.hstack([X_numeric,X_categorical.todense()])

y_pred = model.predict(X_test)


submission = pd.DataFrame({"PassengerId": titanic["PassengerId"],
                           "Survived": y_pred})
submission.to_csv("submission.csv", index=False)
