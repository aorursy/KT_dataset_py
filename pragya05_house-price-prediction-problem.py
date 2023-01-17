# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.columns
train.SalePrice.describe()
# Correlation Matrix (Heatmap Style)
corr_mat = train.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corr_mat, vmax = 0.8)
# "SalePrice" correlation matrix (Zoomed Heatmap style)
cols = corr_mat.nlargest(10, 'SalePrice')['SalePrice'].index
top_10_corr_mat = corr_mat.loc[cols, cols]
sns.heatmap(top_10_corr_mat, vmax = 0.8, annot = True)
train.columns
# Scatterplot
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
# Outliar treatment
# Bivariate Analysis
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)
data.plot.scatter(x = 'GrLivArea', y = 'SalePrice', ylim = (0, 800000))
# Deleting large GrLivArea values
train.sort_values(by = 'GrLivArea', ascending = False)[ : 2]
train = train.drop([1298, 523])
# After outliar removal from GrLivArea
data = pd.concat([train['GrLivArea'], train['SalePrice']], axis = 1)
data.plot.scatter(x = 'GrLivArea', y = 'SalePrice', ylim = (0, 800000))
# Outliar treatment
# Bivariate Analysis
data = pd.concat([train['TotalBsmtSF'], train['SalePrice']], axis = 1)
data.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice', ylim = (0, 800000))
# Do not remove outliars since they follow the Trend
# Histogram and Normal Probability Plot for 'SalePrice'
sns.distplot(train['SalePrice'], fit = stats.norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
# 'SalePrice' is not normal. To make it normal take log transform of 'SalePrice'
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit = stats.norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
# Histogram and Normal Probability Plot for 'GrLivArea'
sns.distplot(train['GrLivArea'], fit =stats.norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot = plt)
# 'GrLivArea' is not normal. To make it normal take log tranform of 'GrLivArea'
train['GrLivArea'] = np.log(train['GrLivArea'])
sns.distplot(train['GrLivArea'], fit = stats.norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot = plt)
# Histogram and Normal Probablity Plot for 'TotalBsmtSF'
sns.distplot(train['TotalBsmtSF'], fit = stats.norm)
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot = plt)
# 'TotalBsmtSF' has positive skewness 
# Lot of zero values, log transform for variables with zero values cannot be done
# Only log transform where 'TotalBsmtSF' > 0
train['TotalBsmtSF'] = np.log(train.loc[train.TotalBsmtSF > 0, 'TotalBsmtSF'])
train.TotalBsmtSF.fillna(0, inplace = True)
# Plot histogram and probability plot for 'TotalBsmtSF' after log transform
sns.distplot(train.loc[train['TotalBsmtSF'] > 0, 'TotalBsmtSF'] , fit = stats.norm)
fig = plt.figure()
res = stats.probplot(train.loc[train['TotalBsmtSF'] > 0, 'TotalBsmtSF'], plot = plt)
# Split dataset into dependent and independent variables
X = train.loc[:, ['TotalBsmtSF', 'GrLivArea', 'FullBath', 'OverallQual', 'GarageCars']]
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Fitting multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# Summary of linear regression model
mean_squared_error(y_test, y_pred)
# Transform test data
test['GrLivArea'] = np.log(test['GrLivArea'])
test['TotalBsmtSF'] = np.log(test.loc[test.TotalBsmtSF > 0, 'TotalBsmtSF'])
test.TotalBsmtSF.fillna(0, inplace = True)
test_subset = test.loc[:, ['TotalBsmtSF', 'GrLivArea', 'FullBath', 'OverallQual', 'GarageCars']]
test_subset.fillna(0, inplace = True)
y_pred_test = regressor.predict(test_subset)
# Preparing submission
submission = pd.concat([test['Id'], pd.Series(y_pred_test)], axis = 1)
submission.columns = ['Id','SalePrice']
submission.to_csv('xyz.csv',index=False)