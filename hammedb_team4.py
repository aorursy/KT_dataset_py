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
data = pd.read_csv("../input/train_technidus.csv")
test = pd.read_csv("../input/test_technidus.csv")
data.head()
test.head()
data.info()
data.describe()
test.describe()
data["Monthlyincome"] = np.round(data.YearlyIncome / 12)
test["Monthlyincome"] = np.round(test.YearlyIncome / 12)

y = data.AveMonthSpend
from  sklearn.preprocessing import LabelEncoder, OneHotEncoder

encode = LabelEncoder()
# onehot = OneHotEncoder(sparse=False ) 
data["MaritalStatus"] = encode.fit_transform(data['MaritalStatus'])
data["Gender"] = encode.fit_transform(data['Gender'])
test["MaritalStatus"] = encode.fit_transform(test['MaritalStatus'])
test["Gender"] = encode.fit_transform(test['Gender'])
# data["Occupation"] = onehot.fit_transform(data['Occupation']).reshape(1, -1)
data = pd.concat([data, pd.get_dummies(data.Occupation, prefix="Occupation")], axis=1)
data = pd.concat([data, pd.get_dummies(data.Education, prefix="Education")], axis=1)
# data = pd.concat([data, pd.get_dummies(data.CountryRegionName, prefix="CountryRegionName")], axis=1)
test = pd.concat([test, pd.get_dummies(test.Occupation, prefix="Occupation")], axis=1)
test = pd.concat([test, pd.get_dummies(test.Education, prefix="Education")], axis=1)
# test = pd.concat([test, pd.get_dummies(test.CountryRegionName, prefix="CountryRegionName")], axis=1)
data.drop(["Title", "FirstName", "MiddleName", "LastName", "Suffix","StateProvinceName","City", "Education", "Occupation", "CountryRegionName","BirthDate", "AddressLine1", "AddressLine2", "PostalCode", "PhoneNumber","AveMonthSpend" ], axis=1, inplace=True)
test.drop(["Title", "FirstName", "MiddleName", "LastName", "Suffix","StateProvinceName","City", "Education", "Occupation", "CountryRegionName","BirthDate", "AddressLine1", "AddressLine2", "PostalCode", "PhoneNumber","AveMonthSpend" ], axis=1, inplace=True)
data.describe()
test.describe()
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# X = scale.fit_transform(data)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
from sklearn.tree import DecisionTreeRegressor
tree =DecisionTreeRegressor()
tree.fit(data, y)
y_pred = tree.predict(test)
output = pd.DataFrame({'CustomerID': test.CustomerID,
                       'AveMonthSpend': y_pred })
print(output)
output.to_csv('submission.csv', index=False)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex6 import *
print("\nSetup complete")

