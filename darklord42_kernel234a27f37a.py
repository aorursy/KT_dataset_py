# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# reading the data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape, test.shape
train.head()
train.info()
plt.figure(figsize=(12, 10))
sns.heatmap(train.corr())
plt.figure(figsize=(12, 10))
sns.heatmap(train[train.corr().nlargest(10, 'SalePrice')['SalePrice'].index].corr())
sns.scatterplot(train['OverallQual'],train['SalePrice'])
sns.scatterplot(train['TotalBsmtSF'],train['SalePrice'])
train = train.drop(train[train['TotalBsmtSF'] > 6000].index)
sns.scatterplot(train['GarageArea'],train['SalePrice'])
sns.scatterplot(train['GrLivArea'],train['SalePrice'])
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 250000)].index)
train['SalePrice'].plot.hist(bins=30)
train['SalePrice'] = np.log1p(train['SalePrice'])
train['SalePrice'].plot.hist(bins=30)
train.shape
all_data = pd.concat([train.drop('SalePrice', axis=1), test]).drop('Id', axis=1)
all_data
for col in all_data:
    if all_data[col].dtype == 'object':
        all_data[col].fillna(value='No Data', inplace=True)
    else:
        all_data[col].fillna(all_data[col].mean(), inplace=True)
all_data = pd.get_dummies(all_data)
train_new = all_data.iloc[:1458,:]
test_new = all_data.iloc[1458:,:]
X_train, X_test, y_train, y_test = train_test_split(train_new, train['SalePrice'], random_state=50, test_size=.30)
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
np.sqrt(mean_squared_log_error(y_test, y_pred_linreg)), r2_score(y_test, y_pred_linreg)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
np.sqrt(mean_squared_log_error(y_test, y_pred_dt)), r2_score(y_test, y_pred_dt)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
np.sqrt(mean_squared_log_error(y_test, y_pred_rf)), r2_score(y_test, y_pred_rf)
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
np.sqrt(mean_squared_log_error(y_test, y_pred_svr)), r2_score(y_test, y_pred_svr)
from xgboost.sklearn import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
np.sqrt(mean_squared_log_error(y_test, y_pred_xgb)), r2_score(y_test, y_pred_xgb)
sub_linreg = pd.DataFrame({'Id':test.Id, 'SalePrice':np.exp(linreg.predict(test_new)) - 1})
sub_linreg.to_csv('sub_linreg.csv', index=False)
