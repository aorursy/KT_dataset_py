# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPRegressor as mlp 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

housing = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
# housing.head
# housing.info()
housing['longitude'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.figure(figsize=(10000,10000))
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population", figsize=(15,8),
        c="median_house_value", cmap=plt.get_cmap("jet"),
    )

# housing.plot(kind="line", x="longitude", y="latitude", alpha=0.4,
#          label="population", figsize=(15,8),
#         c="median_house_value", cmap=plt.get_cmap("jet"),
#     )
# plt.legend

housing.dropna(inplace=True)

housing = pd.get_dummies(data=housing,columns = ['ocean_proximity'])
housing['ocean_proximity'] = housing['ocean_proximity_<1H OCEAN']*1+ housing['ocean_proximity_INLAND']*2+ housing['ocean_proximity_ISLAND']*3+ housing['ocean_proximity_NEAR BAY']*4+housing['ocean_proximity_NEAR OCEAN']*5
housing = housing.drop(['ocean_proximity_INLAND','ocean_proximity_<1H OCEAN','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN'], axis =1)

housing

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
housing = (housing-housing.min())/(housing.max()-housing.min())
housing.describe()
corr = housing.corr()
corr

corr
from sklearn.model_selection import train_test_split


set_X = housing.drop('median_house_value',axis= 1)
set_Y = housing['median_house_value']
train_set_X , test_set_X, train_set_Y , test_set_Y = train_test_split(set_X,set_Y , test_size = 0.2 , random_state = 69)

from sklearn.linear_model import LinearRegression as lr_model


# model = lr_model()
model = mlp()
model.fit(train_set_X,train_set_Y)


print(f'train score:{model.score(train_set_X,train_set_Y)}, test score:{model.score(test_set_X,test_set_Y)}')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 133,
                           n_jobs = 32,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)
rf.fit(train_set_X, train_set_Y)


print(f'train score:{rf.score(train_set_X,train_set_Y)}, test score:{rf.score(test_set_X,test_set_Y)}')