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
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
train = pd.read_csv('../input/train.csv')
train.head()
train.hist(bins=50, figsize=(20,15))
plt.show()
corr_matrix = train.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'SalePrice']
scatter_matrix(train[attributes], figsize=(12,8))
plt.show()
train.plot(kind='scatter', x='OverallQual', y='SalePrice', alpha=0.2, figsize=(10,8))
plt.show()
fig = plt.subplots(figsize=(10,8))
sns.boxplot(x='OverallQual', y='SalePrice', data=train)
plt.show()
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
df_t, df_v = train_test_split(train, test_size=0.2, random_state=0)
df_v['Avg_Qual'] = 0
for qual in df_v['OverallQual'].unique():
    q_mean = df_t.loc[df_t['OverallQual'] == qual,'SalePrice'].mean()
    df_v.loc[df_v['OverallQual'] == qual, 'Avg_Qual'] = q_mean
def rmse_score(data, pred):
    rmse_score = np.sqrt(mean_squared_error(np.log(data), np.log(pred)))
    print ("RMSE Score %.2f" %rmse_score)
rmse_score(df_v['SalePrice'], df_v['Avg_Qual'])
test = pd.read_csv('../input/test.csv')
test.head()
fig = plt.subplots(figsize=(10,8))
sns.distplot(test['OverallQual'],kde=False)
plt.show()
test['SalePrice'] = 0
for qual in test['OverallQual'].unique():
    q_mean = df_t.loc[df_t['OverallQual'] == qual,'SalePrice'].mean()
    test.loc[test['OverallQual'] == qual, 'SalePrice'] = q_mean
submission = test.loc[:, ['Id', 'SalePrice']]
#submission.to_csv('accountant_house1.csv', index=False)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print (train.shape)
print (test.shape)
print (train['Id'].describe(), test['Id'].describe())
test['SalePrice'] = 0
train['Source'] = 'train'
test['Source'] = 'test'
print (train.shape)
print (test.shape)
data_tot = train.append(test)
print (data_tot.shape)
copy_data = data_tot.copy()
obj_col = copy_data.dtypes[copy_data.dtypes == 'object']
nobj_col = copy_data.dtypes[copy_data.dtypes != 'object']
print (obj_col.index)
print (nobj_col.index)
copy_data = pd.get_dummies(copy_data)
obj_col = copy_data.dtypes[copy_data.dtypes == 'object']
nobj_col = copy_data.dtypes[copy_data.dtypes != 'object']
print (obj_col.index)
print (nobj_col.index)
copy_null = copy_data[nobj_col.index].isnull().sum()
print (copy_null[copy_null != 0])
median_fill = copy_data[nobj_col.index].median()
copy_data = copy_data[nobj_col.index].fillna(median_fill)
copy_null = copy_data.isnull().sum()
print (copy_null[copy_null != 0])
copy_train = copy_data[copy_data['Source_train'] == 1]
mod_t, mod_v = train_test_split(copy_train, test_size=0.2, random_state=0)
X = mod_t.drop('SalePrice', axis=1)
y = mod_t['SalePrice']
X_v = mod_v.drop('SalePrice', axis=1)
y_v = mod_v['SalePrice']
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X_v)
rmse_score(y_v, np.abs(y_pred))
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.01, solver='cholesky')
ridge_reg.fit(X, y)
y_pred = ridge_reg.predict(X_v)
rmse_score(y_v, np.abs(y_pred))
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X, y)
y_pred = lasso_reg.predict(X_v)
rmse_score(y_v, np.abs(y_pred))
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.03, l1_ratio=0.85)
elastic_net.fit(X, y)
y_pred = elastic_net.predict(X_v)
rmse_score(y_v, np.abs(y_pred))
copy_test = copy_data[copy_data['Source_test'] == 1]
copy_test = copy_test.drop('SalePrice', axis=1)
y_test = elastic_net.predict(copy_test)
copy_test['SalePrice'] = y_test
submission = copy_test.loc[:, ['Id', 'SalePrice']]
#submission.to_csv('accountant_house2.csv', index = False)
from sklearn.model_selection import GridSearchCV
param_grid = [{'alpha': np.arange(0.01, 0.2, 0.01),
             'l1_ratio': np.arange(0, 1, 0.1)}]
elastic = ElasticNet()
grid_search = GridSearchCV(elastic, param_grid, cv=5,
                           scoring='mean_squared_error')
grid_search.fit(X, y)
grid_search.best_params_
grid_search.best_estimator_
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.06, l1_ratio=0.9)
elastic_net.fit(X, y)
y_pred = elastic_net.predict(X_v)
rmse_score(y_v, np.abs(y_pred))
copy_test = copy_data[copy_data['Source_test'] == 1]
copy_test = copy_test.drop('SalePrice', axis=1)
y_test = elastic_net.predict(copy_test)
copy_test['SalePrice'] = y_test
submission = copy_test.loc[:, ['Id', 'SalePrice']]
submission.to_csv('accountant_house3.csv', index = False)
