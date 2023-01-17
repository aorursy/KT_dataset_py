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
#libs
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


#datasets
train = pd.read_csv("~/train.csv")
test = pd.read_csv("~/test.csv")

#################################

#fields w/ NaNs
train.dtypes[lambda x : [ x for x in train if train[x].isnull().any() ]]

#numeric fields
T = train._get_numeric_data()

#alpha fields
A = train.select_dtypes(include=['object'])

#impute NaNs as avg
Tr_i = pd.DataFrame(Imputer().fit_transform(T))
Tr_i.columns = T.columns
#T_i = T.drop( [ x for x in T.columns if train[x].isnull().any() ], axis=1) #Drop NaN columns

#y = SalePrice
y = Tr_i.SalePrice

#log transform skewness
Tr_i = np.log1p(Tr_i)

#dummy categories
A_i = pd.get_dummies(A)

##########################

#fields w/ NaNs
test.dtypes[lambda x : [ x for x in test if test[x].isnull().any() ]]

#numeric fields
T = test._get_numeric_data()

#alpha fields
A = test.select_dtypes(include=['object'])

#impute NaNs as avg
Te_i = pd.DataFrame(Imputer().fit_transform(T))
Te_i.columns = T.columns
#T_i = T.drop( [ x for x in T.columns if train[x].isnull().any() ], axis=1) #Drop NaN columns

#log transform skewness
Te_i = np.log1p(Te_i)

#dummy categories
B_i = pd.get_dummies(A)

#################################

#align train and text cat vars
A_ii, B_ii = A_i.align(B_i, join='inner', axis=1)

#re-concat train variables
X = Tr_i.iloc[:, 1:len(Tr_i.columns)-1]
X = pd.concat([X, A_ii], axis=1)

#Cross-validation Batching
train_X, val_X, train_y, val_y = train_test_split(X, y)#,random_state = 0)

#Random Forest validation
P = RandomForestRegressor().fit(train_X, train_y)
p = P.predict(val_X)
print(mean_squared_error(np.log(val_y), np.log(p))**0.5)

#re-concat test variables
X = Te_i.iloc[:, 1:len(Te_i.columns)]
X = pd.concat([X, B_ii], axis=1)

#Random Forest test
q = P.predict(X)

###################################

#submit
pd.DataFrame({'Id': test.Id, 'SalePrice': q}).to_csv('out.csv', index=False)