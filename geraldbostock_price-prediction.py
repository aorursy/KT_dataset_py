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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
train_data = pd.read_csv('../input/train.csv')

train_numeric = train_data.select_dtypes(exclude=['object']).columns
train_data.describe()
train_num_corr = train_data[train_numeric].drop(['Id'], axis=1)



cmap = sns.cubehelix_palette(light = 0.95, as_cmap = True)

sns.set(font_scale=1.2)

plt.figure(figsize = (11, 11))

sns.heatmap(abs(train_num_corr.corr(method = 'pearson')), vmin = 0, vmax = 1, square = True, cmap = cmap);
var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'GarageArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
for col_name in train_data.columns:

    if(train_data[col_name].dtype == 'object'):

        train_data[col_name] = train_data[col_name].astype('category')

        train_data[col_name] = train_data[col_name].cat.codes
total_missing_values = train_data.isnull().sum().sort_values(ascending=False)

missing_value_percentages = (total_missing_values / train_data.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total_missing_values, missing_value_percentages], axis=1, keys=['Total', 'Percent'], sort=False)



train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)

train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
X = train_data[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']]

y = train_data['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = LinearRegression()

reg.fit(X_train, y_train)



y_pred = reg.predict(X_test)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



print('Mean absolute error: ' + str(mean_absolute_error(y_test, y_pred)))

print('Mean squared error: ' + str(mean_squared_error(y_test, y_pred)))

print('r2 score: ' + str(r2_score(y_test, y_pred)))
test_data = pd.read_csv('../input/test.csv')



for col_name in test_data.columns:

    if(test_data[col_name].dtype == 'object'):

        test_data[col_name] = test_data[col_name].astype('category')

        test_data[col_name] = test_data[col_name].cat.codes

        

test_total_missing_values = test_data.isnull().sum().sort_values(ascending=False)

test_missing_value_percentages = (total_missing_values / test_data.isnull().count()).sort_values(ascending=False)



test_missing_data = pd.concat([test_total_missing_values, test_missing_value_percentages], axis=1, keys=['Total', 'Percent'], sort=False)



test_data = test_data.drop((test_missing_data[test_missing_data['Total'] > 1]).index,1)

test_data = test_data.drop(test_data.loc[test_data['Electrical'].isnull()].index)

test_data.isnull().sum().max()



test_data.fillna((test_data.mean().round()), inplace=True)
data = test_data[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']]



pred = reg.predict(data)



result_data = {'Id': test_data.Id, 'SalePrice': pred}



result = pd.DataFrame(data = result_data)

result