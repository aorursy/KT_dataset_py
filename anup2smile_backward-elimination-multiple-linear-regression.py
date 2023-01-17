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
## Importing the libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
##Importing the dataset
dataset = pd.read_csv('../input/50_Startups.csv')
dataset.head()
#Segregating the feature variables and dependent variable
X = dataset.iloc[:, :4]

y = dataset.iloc[:, 4:]



X
##Encoding the categorical feature witht the 'Dummy Encoding'
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_transformer



col_trans = make_column_transformer((OneHotEncoder(categories = 'auto'), [3]), remainder = 'passthrough' )



X = col_trans.fit_transform(X)

X
##Avoiding the Dummy Trap
X = X[:, 1:]



X
##Splitting the data into train and test
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )
##Building the regressor for MLR
from sklearn.linear_model import LinearRegression



regressor = LinearRegression().fit(X_train, y_train)



y_predict = regressor.predict(X_test)

y_predict, y_test
##Building the optimal model of MLR with Backward elimination method
import statsmodels.api as sm
X = sm.add_constant(X)
X
y
X_opt = X[:, [0,1,2,3,4,5]]
X_opt
Regressor_opt = sm.OLS(y, X_opt).fit()
Regressor_opt.summary()
X_opt = X[:, [0,1,3,4,5]]
Regressor_opt = sm.OLS(y, X_opt).fit()

Regressor_opt.summary()
X_opt = X[:, [0,3,4,5]]

Regressor_opt = sm.OLS(y, X_opt).fit()

Regressor_opt.summary()
X_opt = X[:, [0,3,5]]

Regressor_opt = sm.OLS(y, X_opt).fit()

Regressor_opt.summary()
X_opt = X[:, [0,3]]

Regressor_opt = sm.OLS(y, X_opt).fit()

Regressor_opt.summary()
y
pd.DataFrame(Regressor_opt.predict(X_opt))