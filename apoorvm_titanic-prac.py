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
titanic_file_path = '../input/train.csv'

titanic_data = pd.read_csv(titanic_file_path)

titanic_data.describe()
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

#data_with_imputed_values = my_imputer.fit_transform(titanic_data)
titanic_data.columns
y = titanic_data.Survived

X = titanic_data.drop(['Survived', 'Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
X.head()
y.head()
X.dtypes.sample(7)
one_hot_encoded_X = pd.get_dummies(X)
one_hot_encoded_X.head()
from sklearn import preprocessing

# Create scaler

scaler = preprocessing.StandardScaler()
standardized_X = scaler.fit_transform(one_hot_encoded_X)
standardized_X
data_with_imputed_values_X = my_imputer.fit_transform(standardized_X)
#standardized_y = scaler.fit_transform(y)

#y = y.reshape(-1,1)
from sklearn.tree import DecisionTreeRegressor
#Define model

titanic_model = DecisionTreeRegressor()
#Fit model

titanic_model.fit(data_with_imputed_values_X, y)
test_file_path = '../input/test.csv'
test_data = pd.read_csv(test_file_path)

test_data

test_X = test_data.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)

test_X.head()
test_X.dtypes.sample(7)
one_hot_encoded_test_X = pd.get_dummies(test_X)

one_hot_encoded_test_X.head()
standardized_test_X = scaler.fit_transform(one_hot_encoded_test_X)
data_with_imputed_values_test_X = my_imputer.fit_transform(standardized_test_X)

data_with_imputed_values_test_X
import math
val_predictions = titanic_model.predict(data_with_imputed_values_test_X)

val_predictions = val_predictions.astype(int)

val_predictions

predicted = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':val_predictions})
predicted
predicted.to_csv('Titanic_predict.csv', encoding='utf-8', index=False)