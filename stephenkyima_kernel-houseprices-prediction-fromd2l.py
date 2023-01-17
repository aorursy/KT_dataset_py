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
%matplotlib inline

import matplotlib.pyplot as plt

from pandas import DataFrame

from sklearn import ensemble

from sklearn.model_selection import train_test_split





path_train = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

path_test = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

data_train = pd.read_csv(path_train)

data_test = pd.read_csv(path_test)

# data_train.describe()
data_train.iloc[0:5, [0,1,2,3,4,-3,-2,-1]]
data_all_features = pd.concat((data_train.iloc[:,1:-1], data_test.iloc[:,1:-1]), sort=False)
numeric_features = data_all_features.dtypes[data_all_features.dtypes != 'object'].index



nonnum_features = data_all_features.dtypes[data_all_features.dtypes == 'object'].index



data_all_features[numeric_features] = data_all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std())).fillna(0)
all_data_afterfixed = pd.get_dummies(data_all_features)

all_data_afterfixed.shape



n_train, n_test = data_train.shape[0], data_test.shape[0]

(n_train, n_test)



all_data_afterfixed.dtypes



x_train = all_data_afterfixed[:n_train]

x_train.shape



x_train.head()



x_train = all_data_afterfixed[:n_train].values

x_train.shape

type(x_train)
all_data_afterfixed.head()
X_train, X_test, y_train, y_test = train_test_split(x_train, data_train['SalePrice'], test_size=0.2)

model_randomforest_regressor = ensemble.RandomForestRegressor(n_estimators=20) 
model_randomforest_regressor.fit(X_train, y_train)

score = model_randomforest_regressor.score(X_test, y_test)

y_pred = model_randomforest_regressor.predict(X_test)

plt.figure()

plt.xlabel('true sales value')

plt.ylabel('predict sales value')

plt.scatter(y_test, y_pred, c = 'green')

plt.title("True value vs predicted value : Linear Regression") 

plt.legend()

plt.show()

print('model accuracy: %f' %score)