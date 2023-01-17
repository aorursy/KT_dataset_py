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
house_price_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

house_price_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print(house_price_train.shape)

house_price_train.head()
house_price_train.describe()
house_price_train.describe(exclude='number')
#欠損値ありリスト

a = house_price_train.isnull().sum()

a[a != 0]
#相関ヒートマップ（色薄いところが相関高い ※数値項目のみ対象）

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(20, 10))

sns.heatmap(house_price_train.select_dtypes(include='number').corr(),vmax=1, vmin=-1, center=0, annot=False)
#相関係数値ソート

ret = np.empty(2)

sp = pd.Series(house_price_train['SalePrice'])

for i in house_price_train.select_dtypes(include='number').columns:

    ret = np.vstack((ret, [i, sp.corr(pd.Series(house_price_train[i]))]))

    

ret[np.argsort(ret[:, 1])[::-1]]
#とりあえず相関>0.5のメンバー

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']

house_price_train[features].describe()
house_price_train[features].dtypes
house_price_train_objlist = house_price_train.select_dtypes(exclude='number').columns



#number以外の項目の確認：ユニーク値が少ない項目のチラ見

for i in house_price_train_objlist:

    if house_price_train[i].nunique() <= 10:

        print(i, house_price_train[i].unique())

        
#number以外の項目の確認：ユニーク値が多い項目のチラ見

for i in house_price_train_objlist:

    if house_price_train[i].nunique() > 10:

        print(i, house_price_train[i].nunique(), house_price_train[i].unique()[0:5], "...")
#category化

house_price_train['flag'] = 0

house_price_test['flag'] = 1

house_price_test['SalePrice'] = 0

house_price_all = pd.concat([house_price_train, house_price_test])



for i in house_price_train_objlist:

    house_price_all[i] = house_price_all[i].astype('category')

    house_price_all[i] = house_price_all[i].cat.codes



house_price_all[house_price_train_objlist].head()
#分離

house_price_train_c = house_price_all[house_price_all['flag'] == 0]

house_price_test_c = house_price_all[house_price_all['flag'] == 1]

house_price_test_c.drop('SalePrice', axis=1, inplace=True)

print(house_price_train_c.shape)

print(house_price_test_c.shape)
#相関係数値ソート(number以外の項目)

ret = np.empty(2)

sp = pd.Series(house_price_train_c['SalePrice'])

for i in house_price_train_objlist:

    ret = np.vstack((ret, [i, sp.corr(pd.Series(house_price_train_c[i]))]))

    

ret[np.argsort(ret[:, 1])[::-1]]
#相関>0.5(絶対値)のメンバー

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',

            'YearBuilt', 'YearRemodAdd', 'Foundation', 'FireplaceQu', 'ExterQual', 'KitchenQual']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(house_price_train_c[features], house_price_train_c['SalePrice'], test_size=0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error



model_rf = RandomForestRegressor()

model_rf.fit(x_train, y_train)

preds = model_rf.predict(x_test)

print(np.sqrt(mean_squared_log_error(y_test, preds)))
#各項目毎の重要性

plt.figure(figsize=(20, 5))

plt.ylim([0, 1])

y = model_rf.feature_importances_

x = np.arange(len(y))

plt.bar(x, y, align="center")

plt.xticks(x, house_price_train_c[features].columns)

plt.show()
print(house_price_test_c[features].dtypes)

house_price_test_c[features].describe()
house_price_test_c['GarageCars'].fillna(0, inplace=True)

house_price_test_c['GarageArea'].fillna(0, inplace=True)

house_price_test_c['TotalBsmtSF'].fillna(0, inplace=True)

house_price_test_c[features].describe()
import math

house_price_test_c['GarageCars'] = house_price_test_c['GarageCars'].apply(lambda x: math.ceil(x))

house_price_test_c['GarageArea'] = house_price_test_c['GarageArea'].apply(lambda x: math.ceil(x))

house_price_test_c['TotalBsmtSF'] = house_price_test_c['TotalBsmtSF'].apply(lambda x: math.ceil(x))

house_price_test_c[features].dtypes
from sklearn.ensemble import GradientBoostingRegressor

model_gr = GradientBoostingRegressor()

model_gr.fit(x_train, y_train)

preds = model_gr.predict(x_test)

print(np.sqrt(mean_squared_log_error(y_test, preds)))
#提出

preds = model_gr.predict(house_price_test_c[features])

output = pd.DataFrame({'Id': house_price_test_c.Id, 'SalePrice': preds})

#output['SalePrice'] = output['SalePrice'].apply(lambda x: math.ceil(x))

output.to_csv('my_submission.csv', index=False)