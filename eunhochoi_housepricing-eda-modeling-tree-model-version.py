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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
pd.options.display.max_columns = 100
train.head()
plt.figure(figsize = (10,5))
sns.distplot(train['SalePrice'])
plt.figure(figsize = (10,5))
sns.boxplot(x=train['SalePrice'])
# Outliar
aaa = train[train['SalePrice'] > 700000]
aaa.head()
train['GrLivArea'].describe()
train['TotalBsmtSF'].describe()
train = train[train['SalePrice'] < 700000]
train = train.drop([train.index[1298] , train.index[533]])
#alldata = pd.concat([train, test])
from sklearn.preprocessing import StandardScaler

#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:5]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-5:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
#scatterplot
sns.set()
cols = ['SalePrice', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','LotArea','GrLivArea','LowQualFinSF', 'TotRmsAbvGrd']
sns.pairplot(train[cols], size = 2)
plt.show();
data = train[['SalePrice', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','LotArea', '1stFlrSF', '2ndFlrSF' ,'GrLivArea', 
                'LowQualFinSF', 'TotRmsAbvGrd', 'YrSold', 'GarageArea']]
plt.figure(figsize=(10,10))
sns.heatmap(data = data.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')
Area = train[['GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']]
Area.head()
plt.figure(figsize = (10,5))
plt.title('Area')
sns.scatterplot(train['1stFlrSF'], train['SalePrice'])
sns.scatterplot(train['2ndFlrSF'], train['SalePrice'])
plt.xlabel('Area')
# Make a relationship between GrLivArea and 1stFlrSF
#alldata['Ratio_1st'] = alldata['1stFlrSF'] / alldata['GrLivArea']
plt.figure(figsize = (6,3))
sns.boxplot(x=train['GrLivArea'])
plt.figure(figsize = (6,3))
sns.boxplot(np.log(train['GrLivArea']))
aaaa = train[np.log(train['GrLivArea']) > 8.5]
bbbb = train[np.log(train['GrLivArea']) < 6.0]
cccc = pd.concat([aaaa, bbbb])
cccc.head()
plt.figure(figsize = (6,3))
sns.boxplot(x=train['1stFlrSF'])
plt.figure(figsize = (6,3))
sns.boxplot(np.log(train['1stFlrSF']))
eeee = train[np.log(train['1stFlrSF']) > 8.2]
eeee.head()
# Interaction Columns
#alldata['Period_Original'] = alldata['YrSold'] - alldata['YearBuilt']
#alldata['Period_Remod'] = alldata['YrSold'] - alldata['YearRemodAdd']
#plt.figure(figsize = (10,5))
#sns.scatterplot(alldata['Period_Remod'], alldata['SalePrice'])
#sns.scatterplot(alldata['Period_Original'], alldata['SalePrice'])
Bsmt = train[['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']]
Bsmt.head()
plt.figure(figsize = (6,3))
sns.boxplot(x=train['TotalBsmtSF'])
plt.figure(figsize = (6,3))
sns.boxplot(np.log(train['TotalBsmtSF']))
dddd = train[np.log(train['TotalBsmtSF']) > 8.5]
dddd.head()
#scatterplot
sns.set()
cols = ['SalePrice', 'TotRmsAbvGrd', 'BedroomAbvGr', 'KitchenAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
sns.pairplot(train[cols], size = 2)
plt.show();
#boxplot
var = 'FullBath'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 4))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=10, ymax=700000);
plt.xticks(rotation=90);
# Bathrooms
#alldata['TotBath'] = alldata['BsmtFullBath'] + alldata['BsmtHalfBath'] + alldata['FullBath'] + alldata['HalfBath']

# Rooms (Baths inclusive)
#alldata['TotRmsAbvGrd_Bath'] = alldata['TotRmsAbvGrd'] + alldata['TotBath']
#boxplot
#var = 'TotRmsAbvGrd_Bath'
#data = pd.concat([train['GrLivArea'], train[var]], axis=1)
#f, ax = plt.subplots(figsize=(8, 4))
#fig = sns.boxplot(x=var, y="GrLivArea", data=data)
#fig.axis(ymin=10, ymax=7000);
#plt.xticks(rotation=90);
#LotShape: General shape of property
#group = alldata.groupby('TotRmsAbvGrd_Bath')['GrLivArea'].agg(['mean'])
#group.columns = ['TotRmsAbvGrd_Bath_GrLivArea']
#group.reset_index(inplace=True)

#alldata = pd.merge(alldata, group, on=('TotRmsAbvGrd_Bath'), how='left')
#alldata['TotRmsAbvGrd_Bath_GrLivArea'] = alldata['TotRmsAbvGrd_Bath_GrLivArea'].astype(np.float16)
#scatterplot
sns.set()
cols = ['SalePrice', 'GarageArea', 'GarageYrBlt', 'GarageCars']
sns.pairplot(train[cols], size = 2)
plt.show();
ccc = train[train['GarageCars'] == 4]
ccc.head()
train = train[train['SalePrice'] < 700000]
train = train.drop([train.index[420], train.index[747], train.index[1190], train.index[1340], train.index[1350]])
train = train.drop([train.index[1298] , train.index[533]])
alldata = pd.concat([train, test])
alldata['Ratio_1st'] = alldata['1stFlrSF'] / alldata['GrLivArea']

alldata['Period_Original'] = alldata['YrSold'] - alldata['YearBuilt']
alldata['Period_Remod'] = alldata['YrSold'] - alldata['YearRemodAdd']


alldata['TotBath'] = alldata['BsmtFullBath'] + alldata['BsmtHalfBath'] + alldata['FullBath'] + alldata['HalfBath']
alldata['TotRmsAbvGrd_Bath'] = alldata['TotRmsAbvGrd'] + alldata['TotBath']
#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)
#scatterplot
sns.set()
cols = ['LotFrontage', 'SalePrice']
sns.pairplot(train[cols], size = 2)
plt.show();
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in alldata.columns[alldata.dtypes == object]:
    alldata[i] = le.fit_transform(list(alldata[i]))
alldata2 = alldata.drop(['Id', 'SalePrice'], axis = 1)
alldata2 = alldata2.fillna(-1)
train2 = alldata2[:len(train)]
test2 = alldata2[len(train):]
from xgboost import XGBRegressor
xgb = XGBRegressor(max_depth = 6, colsample_bytree = 0.7, subsample_bytree = 0.8, n_estimators = 1000, learning_rate = 0.01)
xgb.fit(train2, train['SalePrice'])
result = xgb.predict(test2)
from lightgbm import LGBMRegressor
lgb = LGBMRegressor(num_leaves = 75, colsample_bytree = 0.7, subsample_bytree = 0.8, n_estimators = 1000, learning_rate = 0.01)
lgb.fit(train2, train['SalePrice'])
result2 = lgb.predict(test2)
from catboost import CatBoostRegressor  
cb = CatBoostRegressor()
cb.fit(train2, train['SalePrice'])
result3 = cb.predict(test2)
sub = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
sub['SalePrice'] = result*0.5 + result2*0.25 + result3*0.25
sub.head()
sub.to_csv('House(Tree)v2.csv', index = False)
