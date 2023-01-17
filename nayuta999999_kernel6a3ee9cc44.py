# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(data_test.head())

model = RandomForestRegressor(random_state = 1)

y = data_train.Survived

y = y.fillna(y.mean())

features = ['Pclass', 'Age', 'Fare']

X = data_train[features]

X = X.fillna(X.mean())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model.fit(train_X, train_y)

test_X = data_test[features]

test_X = test_X.fillna(test_X.mean())

test_P = data_test['PassengerId']

print(test_X.head())

"""

for i in range(len(test_X)):

    print("{} {} ".format(i+1, test_P[i], round(model.predict(test_X[i]))) )

"""

print(list(map(round, model.predict(test_X))))