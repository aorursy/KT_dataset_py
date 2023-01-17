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
titanic_train_data_filepath = "../input/train.csv"
titanic_train_data = pd.read_csv(titanic_train_data_filepath)
titanic_train_data.describe()
titanic_train_data.columns
titanic_train_data =  titanic_train_data.dropna(axis = 0)
train_y = titanic_train_data.Survived
titanic_train_data.columns
titanic_features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
train_X = titanic_train_data[titanic_features]
train_X.describe()
train_X.head()
from sklearn.tree import DecisionTreeRegressor
titanic_model = DecisionTreeRegressor(random_state = 1)
titanic_model.fit(train_X, train_y)
titanic_validation_data_filepath = '../input/test.csv'
titanic_validation_data = pd.read_csv(titanic_validation_data_filepath)
titanic_validation_data.describe()
titanic_validation_data = titanic_validation_data.dropna(axis = 0)
val_X = titanic_validation_data[titanic_features]
val_X.describe()
print(titanic_model.predict(val_X))