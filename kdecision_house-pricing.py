# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import norm

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
import matplotlib.pyplot as plt

import seaborn as sns

train.info()
# Lot Area vs SalePrice

# GrLivArea vs SalePrice

#TotalBsmtSF  vs SalePrice

#GarageArea vs SalePrice

#SalePrice vs Overall Quality
plt.figure(figsize=(12,5))

sns.scatterplot(x=train['SalePrice'],y=train['LotArea'])
plt.figure(figsize=(12,5))

sns.scatterplot(x=train['SalePrice'],y=train['GrLivArea'])
plt.figure(figsize=(12,5))

sns.scatterplot(x=train['SalePrice'],y=train['TotalBsmtSF'])
plt.figure(figsize=(12,5))

sns.scatterplot(x=train['SalePrice'],y=train['GarageArea'])
plt.figure(figsize=(12,5))

sns.boxplot(x=train['OverallQual'],y=train['SalePrice'])
plt.figure(figsize=(12,5))

sns.distplot(train['SalePrice'],kde=True,fit=norm)
tx = train['SalePrice'].values

print(tx)
log_norm_saleprice = np.log(train['SalePrice'])
plt.figure(figsize=(12,5))

sns.distplot(log_norm_saleprice,kde=True,fit=norm)
corr = train.corr()

plt.figure(figsize=(12,12))

sns.heatmap(corr)
null_val_count = train.isnull().sum()/len(train)
pd.set_option('display.max_columns', None)
null_val_count = null_val_count.sort_values(ascending=False)
len(null_val_count)
null_val_count[null_val_count!=0]
# to drop PoolQC MiscFeature Alley Fence FireplaceQu
train = train.drop(columns = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
y_train = train['SalePrice'].values
train = train.drop(columns = ['SalePrice','Id'],axis=1)
train.info()
col = train.columns

for c in col:

    if train[c].dtype == 'object':

        train[c] = train[c].replace(np.nan,"Unknown")

    else:

        train[c].fillna(train[c].median(),inplace=True)
train.info()
test_id = test['Id']
test = test.drop(columns=['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
test.info()
col = test.columns

for c in col:

    if test[c].dtype == 'object':

        test[c] = test[c].replace(np.nan,"Unknown")

    else:

        test[c].fillna(test[c].median(),inplace=True)
train_len = train.shape[0]
df = pd.concat((train,test)).reset_index(drop=True)
df.describe()
df.shape
X = pd.get_dummies(df)
X_train = X[:train_len]

X_test = X[train_len:]
from sklearn.ensemble import RandomForestRegressor

rgr = RandomForestRegressor(max_depth=3, random_state=0)
rgr_n = rgr.fit(X_train,y_train)
pre = rgr.predict(X_test)
pre
xgb = XGBRegressor()
xgb.fit(X_train,y_train)

pred = xgb.predict(X_test)
pred
data_pre = pd.DataFrame()
data_pre['Id']=test_id

data_pre['SalePrice'] = pre
data_pre.head()
data_pre.to_csv('submission_xg.csv',index=False)