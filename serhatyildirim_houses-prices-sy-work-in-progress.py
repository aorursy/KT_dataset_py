# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for plotting

from matplotlib import colors

from matplotlib.ticker import PercentFormatter

import seaborn as sns

import missingno as msno

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
test.shape
full = train.append( test, ignore_index = True )
full.head()
full.shape
full.describe()
full.isnull().any().count()
full.dtypes.value_counts()
fig, ax = plt.subplots(figsize=(13,13)) 

ax2 = sns.heatmap(full.isnull(), cbar=False)
msno.heatmap(train)
sns.distplot(train['SalePrice'])
train['SalePrice_log'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice_log'])
x =full.dtypes[full.dtypes != "object"]

x.head()
full.isnull().any()
full = pd.get_dummies(full)

full.head()
full.fillna(full.mean(), inplace=True)

full.isnull().any()
full_train = full[:train.shape[0]]
full_train.shape
y =  full_train.SalePrice
X = full_train.drop(columns=['SalePrice'])
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, train_size=.7)
dt = DecisionTreeRegressor(random_state=1)
dt.fit(train_X, train_y)
val_predictions = dt.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

val_mae
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
rf_val_mae