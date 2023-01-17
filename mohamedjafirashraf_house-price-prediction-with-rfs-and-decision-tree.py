# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

mpl.style.use('seaborn-darkgrid')

from scipy import stats

from scipy.stats import norm, skew

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)

print(test.shape)
features = ['LotArea', 'LotFrontage', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','SalePrice']

train_data = train[features]
missing_data_train = train_data.isnull()





for column in missing_data_train.columns:

    print(column)

    print(missing_data_train[column].value_counts())

    print("")
train_data.dropna(subset=['LotFrontage'], axis=0, inplace=True)

train_data.reset_index(drop=True,inplace=True)
missing_data_train = train_data.isnull()



for column in missing_data_train.columns:

    print(column)

    print(missing_data_train[column].value_counts())

    print("")
train_data.info()
train_data.head()
train_data.describe()
train_data.corr()
corr_train = train_data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corr_train, vmax=0.9, square=True, center = 0, cmap = 'viridis')
sns.regplot(x='LotArea',y='SalePrice',data=train_data)
train_data[['LotArea','SalePrice']].corr()
sns.regplot(x='LotFrontage',y='SalePrice',data=train_data)
train_data[['LotFrontage','SalePrice']].corr()
sns.regplot(x='YearBuilt',y='SalePrice',data=train_data)
train_data[['YearBuilt','SalePrice']].corr()
sns.regplot(x='1stFlrSF',y='SalePrice',data=train_data)
train_data[['1stFlrSF','SalePrice']].corr()
sns.regplot(x='2ndFlrSF',y='SalePrice',data=train_data)
train_data[['2ndFlrSF','SalePrice']].corr()
sns.regplot(x='FullBath',y='SalePrice',data=train_data)
train_data[['FullBath','SalePrice']].corr()
sns.regplot(x='BedroomAbvGr',y='SalePrice',data=train_data)
train_data[['BedroomAbvGr','SalePrice']].corr()
sns.regplot(x='TotRmsAbvGrd',y='SalePrice',data=train_data)
train_data[['TotRmsAbvGrd','SalePrice']].corr()
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X = train_data[features]

y = train_data.SalePrice

#split the data

train_X, test_X, train_y, test_y =train_test_split(X, y,random_state=1)



#Decision Tree Regressor

dtr_model = DecisionTreeRegressor()

dtr_model.fit(train_X, train_y)

val_predictions = dtr_model.predict(test_X)

val_mae = mean_absolute_error(val_predictions, test_y)

print("Validation MAE for Decision tree model: {:,.0f}".format(val_mae))



#Decision Tree Regressor Max Leaf

dtr_model_1 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

dtr_model_1.fit(train_X, train_y)

val_predictions = dtr_model_1.predict(test_X)

val_mae = mean_absolute_error(val_predictions, test_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



#Random Forest Regressor

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(test_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, test_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))