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
from __future__ import division
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import math
from scipy.stats import pearsonr
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.head()
df.shape
df.dtypes.unique()
df.info()
df.describe()
# checking the categorical features
df.select_dtypes(include = ['object']).columns.tolist()
# check any null value
print(df.isnull().any().sum())
print(df.isnull().any(axis = 1).sum())
df.columns
features = df.iloc[:, 3:].columns.tolist()
features
target = df.iloc[:,2].name
target
# correlation
correlations = {}
for f in features:
    data_temp = df[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]
df_correlations = pd.DataFrame(correlations, index = ['Value']).T
df_correlations.loc[df_correlations['Value'].abs().sort_values(ascending = False).index]
# top 5 features are the most correlated features
y = df.loc[:, ['sqft_living', 'grade', target]].sort_values(target, ascending = True).values
x = np.arange(y.shape[0])
%matplotlib inline
plt.subplot(3, 1, 1)
plt.plot(x, y[:, 0])
plt.title('sqft and Grade vs Price')
plt.ylabel('Sqft')

plt.subplot(3,1, 2)
plt.plot(x, y[:, 1])
plt.ylabel('Grade')

plt.subplot(3, 1, 3)
plt.plot(x, y[:, 2])
plt.ylabel('Price')

plt.show()
df.columns
sns.pairplot(df[['sqft_living', 'grade','sqft_above','sqft_living15','bathrooms', 'price' ]])
from sklearn.model_selection import train_test_split
df.columns
X = df[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15']].values
y = df.price.values
df.view.unique()
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, tree, linear_model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
Linearmodel = LinearRegression()
Linearmodel.fit(X_train, y_train)
linear_pred = Linearmodel.predict(X_test)

Linearmodel.score(X_test, y_test)
# calculate the root mean squared error
print('RMSE: %.2f' % math.sqrt(np.mean((linear_pred - y_test)**2)))
# the error is too much so lets try xgboost
xgb = xgboost.XGBRegressor(n_estimators = 100, learning_rate = 0.08, gamma = 0, subsample = 0.75,
                          colsample_bytree = 1, max_depth = 7)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print('xgb RMSE: %.2f' % math.sqrt(np.mean((xgb_pred-y_test)**2)))
print('linear RMSE: %.2f' % math.sqrt(np.mean((linear_pred - y_test)**2)))
from sklearn.metrics import explained_variance_score
print('xgb: ',explained_variance_score(xgb_pred, y_test))
print('linear: ', explained_variance_score(linear_pred, y_test))
df.head()
xgb_ypred = pd.DataFrame(xgb_pred, columns = ['price'])
xgb_ypred.head()
xgb_ypred.to_csv('submission.csv', index = False)
