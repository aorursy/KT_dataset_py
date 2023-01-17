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
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_ytest = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_train.head()
df_test.head()
df_train.columns
df_train.dtypes
df_train['SalePrice'].describe()
df_train.shape

df_test.shape

df_train.plot(kind='scatter', x='Id', y='SalePrice')
df_train = df_train[df_train['SalePrice']<350000]

df_train.plot(kind='scatter', x='Id', y='SalePrice')
sns.distplot(df_train['SalePrice'])
corr = df_train.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr,vmax=.8, square=True)

k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cm, annot=True, vmax = 0.8, yticklabels=cols.values, xticklabels=cols.values, )
cols = cols.drop(['GarageArea','1stFlrSF'])
sns.set()
sns.pairplot(df_train[cols], size=3)
plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
cols_test = cols.drop(['SalePrice'])
df_test[cols_test]
df_test[cols_test].isnull().any()
df_ytest.isnull().any()
df_train = df_train.drop((missing_data[missing_data['Total']>1]).index, 'columns')
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_test['SalePrice'] = df_ytest['SalePrice']
df_test.at[df_test['GarageCars'].isnull().index,'GarageCars'] = df_test.GarageCars.mean()
df_test.at[df_test['TotalBsmtSF'].isnull().index,'TotalBsmtSF'] = df_test.TotalBsmtSF.mean()

df_test[cols].isnull().any()
df_test[cols]
X_train = df_train[cols].drop(['SalePrice'], axis = 1)
y_train = df_train['SalePrice']
X_test = df_test[cols].drop(['SalePrice'], axis = 1)
y_test = df_ytest['SalePrice']
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
lr.intercept_
lr.coef_
coeff_df = pd.DataFrame(lr.coef_,X_train.columns,columns=['Coefficient'])
coeff_df
lr.score(X_test,y_test)
from sklearn import metrics
from math import sqrt
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, y_pred)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, y_pred)))
print('RMSE: {}'.format(sqrt(metrics.mean_squared_error(y_test, y_pred))))
#print('RMSLE: {}'.format(sqrt(metrics.mean_squared_log_error(y_test, y_pred))))
print("R2: {}".format(metrics.r2_score(y_test,y_pred)))
X_test['SalePrice'] = y_test
X_test['SalePricePred'] = y_pred
final_data = pd.concat([df_train[cols], X_test])
final_data
x_ax = 'OverallQual'
ax = final_data.plot.scatter(x=x_ax,y='SalePrice', alpha=0.5)
final_data.plot.scatter(x=x_ax,y='SalePricePred', ax=ax, color='Orange', alpha=0.5)



result = pd.DataFrame() 
result['SalePrice'] = pd.DataFrame(y_pred).iloc[:,0]
result['Id'] = df_ytest['Id']
result
result.to_csv('HPpred3.csv',index=False)