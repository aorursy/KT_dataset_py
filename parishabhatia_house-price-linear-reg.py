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

import sklearn 

import seaborn as sns

import re
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',na_filter=False)

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',na_filter=False)
test.head()
train.head()
corr = train.corr()

cmap = sns.diverging_palette(230,20,as_cmap = True)

f, ax = plt.subplots(figsize=(11, 9))

cax = ax.matshow(corr, cmap='coolwarm')

ticks = np.arange(0,len(corr.columns),1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

plt.xticks(rotation=90)

ax.set_xticklabels(corr.columns)

ax.set_yticklabels(corr.columns)

plt.show()
train.describe()
sns.distplot(train['SalePrice']);
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train[train.columns[1:]].corr()['SalePrice'][:].abs().sort_values(ascending=False)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], height = 2.5)

plt.show();
train_20 = train[['Id','SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]

test_20 = test[['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]
print('train:')

print(train_20.isnull().sum())

print('\ndf_test:')

print(test_20.isnull().sum())
X_train = train_20.drop(columns=['SalePrice','Id'])

y_train = train_20['SalePrice']



X_test = test_20.drop(columns=['Id'])

X_test.fillna(0, inplace=True)
X_train.head()
train['Neighborhood'].value_counts()
dict_neighbor = {

'NAmes'  :{'lat': 42.045830,'lon': -93.620767},

'CollgCr':{'lat': 42.018773,'lon': -93.685543},

'OldTown':{'lat': 42.030152,'lon': -93.614628},

'Edwards':{'lat': 42.021756,'lon': -93.670324},

'Somerst':{'lat': 42.050913,'lon': -93.644629},

'Gilbert':{'lat': 42.060214,'lon': -93.643179},

'NridgHt':{'lat': 42.060357,'lon': -93.655263},

'Sawyer' :{'lat': 42.034446,'lon': -93.666330},

'NWAmes' :{'lat': 42.049381,'lon': -93.634993},

'SawyerW':{'lat': 42.033494,'lon': -93.684085},

'BrkSide':{'lat': 42.032422,'lon': -93.626037},

'Crawfor':{'lat': 42.015189,'lon': -93.644250},

'Mitchel':{'lat': 41.990123,'lon': -93.600964},

'NoRidge':{'lat': 42.051748,'lon': -93.653524},

'Timber' :{'lat': 41.998656,'lon': -93.652534},

'IDOTRR' :{'lat': 42.022012,'lon': -93.622183},

'ClearCr':{'lat': 42.060021,'lon': -93.629193},

'StoneBr':{'lat': 42.060227,'lon': -93.633546},

'SWISU'  :{'lat': 42.022646,'lon': -93.644853}, 

'MeadowV':{'lat': 41.991846,'lon': -93.603460},

'Blmngtn':{'lat': 42.059811,'lon': -93.638990},

'BrDale' :{'lat': 42.052792,'lon': -93.628820},

'Veenker':{'lat': 42.040898,'lon': -93.651502},

'NPkVill':{'lat': 42.049912,'lon': -93.626546},

'Blueste':{'lat': 42.010098,'lon': -93.647269}

}
train['Lat'] = train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])

train['Lon'] = train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])



test['Lat'] = test['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])

test['Lon'] = test['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
train.select_dtypes('object').columns
from sklearn import preprocessing



for columns in train.select_dtypes('object').columns:

    enc = preprocessing.LabelEncoder()

    enc.fit(pd.concat([train[columns].astype(str), test[columns].astype(str)],join='outer',sort=False))

    train[columns] = enc.transform(train[columns])

    test[columns] = enc.transform(test[columns])
for columns in test.select_dtypes('object').columns:

    test[columns] = pd.to_numeric(test[columns],errors='coerce')

    train[columns] = pd.to_numeric(train[columns],errors='coerce')

    

test.fillna(0, inplace=True)
X_train = train.drop(columns=['SalePrice','Id'])

y_train = train['SalePrice']



X_test = test.drop(columns=['Id'])
X_train.head()
X_test.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

#x, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)

#regr = RandomForestRegressor(max_depth=2, random_state=0)

#regr.fit(X, y)

tree = RandomForestRegressor(max_depth = 20, n_estimators = 1000, random_state=0)



tree.fit(X_train, y_train)



y_test = pd.Series(tree.predict(X_test))



final = pd.concat([test['Id'], y_test], axis=1, sort=False)

final = final.rename(columns={0:"SalePrice"})

final.to_csv(r'random_forest.csv', index = False)

print("CSV Made")