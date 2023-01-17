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
train.head()
test = pd.read_csv("../input/test.csv")
test.head()
train.describe()
train_y = train.Survived
train_X = train.drop(['Survived'], axis=1)
train_X.head()
train_X = train.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis=1)
train_X.head()
train_X = train.drop(['PassengerId','Name','Ticket','Fare','Cabin','Survived'], axis=1)
train_X.head()
train_y.head()
test.head()
test_X = test.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis=1)
test_X.head()
train_X.describe()

train_X.columns

test_X.describe()
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
encoded_train_X = pd.get_dummies(train_X)
encoded_train_X.head()
encoded_test_X = pd.get_dummies(test_X)
encoded_test_X.head()
encoded_train_X.describe()
prefinal_train_X, prefinal_test_X = encoded_train_X.align(encoded_test_X, join='left', axis=1)
imputed_train_X = pd.DataFrame(imputer.fit_transform(prefinal_train_X))
imputed_test_X = pd.DataFrame(imputer.fit_transform(prefinal_test_X))
from xgboost import XGBRegressor
imputed_train_X.head()
imputed_test_X.head()

imputed_train_X = imputed_train_X.iloc[:, :-1]
imputed_test_X = imputed_test_X.iloc[:, :-1]
model = XGBRegressor()
model.fit(imputed_train_X, train_y, verbose=False)
preds = model.predict(imputed_test_X)
preds
preds_int = preds.astype(int)
preds_int
test_X.head()
test.head()
output = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Survived': preds_int})
output.to_csv('submission.csv', index=False)
