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
import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = pd.read_csv("../input/train.csv")
X.describe()
y = X.pop("Survived")
y.head()
numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head()
X["Age"].fillna(X.Age.mean(), inplace = True)
X.tail()
X[numeric_variables].head()
model = RandomForestClassifier(n_estimators = 100)
model.fit(X[numeric_variables], y)
test = pd.read_csv("../input/test.csv")
test[numeric_variables].head()
test['Age'].fillna(test.Age.mean(), inplace = True)

test = test[numeric_variables].fillna(test.mean()).copy()
y_pred = model.predict(test[numeric_variables])
y_pred
my_submission = pd.DataFrame({'PassengerId': test["PassengerId"],"Survived":y_pred})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
