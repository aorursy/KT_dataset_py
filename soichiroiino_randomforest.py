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
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
import matplotlib.pyplot as plt

import seaborn as sns



# 相関行列でSalePriceと相関が強い説明変数を見つける

corr = train.corr()

plt.figure(figsize=(8,8))

sns.heatmap(corr)

plt.yticks(rotation=0, size=7)

plt.xticks(rotation=90, size=7)

plt.show()



# Select columns with a correlation > 0.5

rel_vars = corr.SalePrice[(corr.SalePrice > 0.5)]

rel_cols = list(rel_vars.index.values)



corr2 = train[rel_cols].corr()

plt.figure(figsize=(8,8))

hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})

plt.yticks(rotation=0, size=10)

plt.xticks(rotation=90, size=10)

plt.show()
#　SalePriceとの相関が0.5よりも大きい説明変数を選び、散布図を書いてみる



cols_viz = ["OverallQual", "YearBuilt", "YearRemodAdd", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageCars", "GarageArea", "SalePrice"]

sns.pairplot(train[cols_viz], size=2.5)

plt.tight_layout()

plt.show()
import missingno as msno



#　欠損値の数を確認する。

print(train.isnull().sum())

print(test.isnull().sum())



#　欠損値を可視化する。

msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))
# trainデータ・testデータともに、カテゴリ型カラムを数値型カラムに変換する

from sklearn.preprocessing import LabelEncoder



for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))



from sklearn.preprocessing import LabelEncoder



for i in range(test.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
print(train["OverallQual"].isnull().sum())

print(train["YearBuilt"].isnull().sum())

print(train["YearRemodAdd"].isnull().sum())

print(train["1stFlrSF"].isnull().sum())

print(train["GrLivArea"].isnull().sum())

print(train["FullBath"].isnull().sum())

print(train["GarageCars"].isnull().sum())
print(test["OverallQual"].isnull().sum())

print(test["YearBuilt"].isnull().sum())

print(test["YearRemodAdd"].isnull().sum())

print(test["1stFlrSF"].isnull().sum())

print(test["GrLivArea"].isnull().sum())

print(test["FullBath"].isnull().sum())

print(test["GarageCars"].isnull().sum())
# testデータの欠損値を補完する

test['GarageCars'].fillna(test['GarageCars'].mean(skipna=True),inplace=True)
#bivariate analysis saleprice/OverallQual

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# OverallQual

train = train.drop(train[(train['OverallQual']>=10) & (train['SalePrice']<250000)].index)
# YearBuilt

#bivariate analysis saleprice/YearBuilt

var = "YearBuilt"

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#YearBuilt

train = train.drop(train[(train['YearBuilt']>=2000) & (train['SalePrice']<100000)].index)

train = train.drop(train[(train['YearBuilt']<=1900) & (train['SalePrice']>400000)].index)
# YearRemodAdd

#bivariate analysis saleprice/YearRemodAdd

var = 'YearRemodAdd'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# YearRemodAdd

train = train.drop(train[(train['YearRemodAdd']<=1990) & (train['SalePrice']>350000)].index)
# GrLivArea

#bivariate analysis saleprice/GrLivArea

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#GrLivArea

train = train.drop(train[(train["GrLivArea"]>=3000) & (train['SalePrice']<300000)].index)
from sklearn.model_selection import train_test_split



# X,yにデータを代入

X = train.loc[:, ["OverallQual", "YearBuilt", "YearRemodAdd", "1stFlrSF", "GrLivArea", "FullBath", "GarageCars"]].values

y = train["SalePrice"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.4, random_state=1)



from sklearn.ensemble import RandomForestRegressor



# ランダムフォレスト回帰のクラスをインスタンス化

forest = RandomForestRegressor(n_estimators=100, criterion="mse", random_state=1, n_jobs=-1)



forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)

y_test_pred = forest.predict(X_test)
#　MSEを出力

from sklearn.metrics import mean_squared_error

print("MSE train: %.3f, test: %.3f" % (mean_squared_error(y_train, y_train_pred), 

                                       mean_squared_error(y_test, y_test_pred)))



#　R^2の出力

from sklearn.metrics import r2_score

print("R^2 train: %.3f, test: %.3f" % (r2_score(y_train, y_train_pred), 

                                       r2_score(y_test, y_test_pred)))
sub = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
Pred = forest.predict(test.loc[:, ["OverallQual", "YearBuilt", "YearRemodAdd", "1stFlrSF", "GrLivArea", "FullBath", "GarageCars"]].values)
sub['SalePrice'] = list(map(int, Pred))
sub.to_csv('submission.csv', index=False)