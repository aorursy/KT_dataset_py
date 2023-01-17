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
# ライブラリのインポート

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression



%matplotlib inline 
# 訓練データの読み込み

df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")



# テストデータの読込

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
# 読み込んだデータの確認

df
# 読み込んだデータの確認

df_test
# カラムの一覧表示

df.columns
# 基本統計量の表示

df.SalePrice.describe()
# ヒストグラムの表示

sns.distplot(df.SalePrice)
#SalePriceの大きい順

df['SalePrice'].sort_values(ascending=False).head()
#正規分布に近づける



y_train = np.log(df['SalePrice'])



sns.distplot(y_train)

plt.show()
y_train
#相関係数を算出

corrmat = df.corr()

corrmat
#正規分布にする前のSalePriseとの相関をみる

# ヒートマップに表示させるカラムの数

k = 10



# SalesPriceとの相関が大きい上位10個のカラム名を取得

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index



# SalesPriceとの相関が大きい上位10個のカラムを対象に相関を算出

cm = np.corrcoef(df[cols].values.T)



# ヒートマップのフォントサイズを指定

sns.set(font_scale=1.25)



# 算出した相関データをヒートマップで表示

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#相関が強く、かつ、変数の意味から必要でありそうな変数の散布図の表示

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'WoodDeckSF', 'HalfBath', 'TotRmsAbvGrd']

sns.pairplot(df[cols], size = 2.5)

plt.show()
#大きく外れている値が2つあるので確認する

df.sort_values(by = 'GrLivArea', ascending = False)[:2]
#大きい値を削除

Xmat = df

Xmat['SalePrice'] = y_train

Xmat = Xmat.drop(index = df[df['Id'] == 1299].index)

Xmat = Xmat.drop(index = df[df['Id'] == 524].index)



y_train = Xmat['SalePrice']

df = Xmat.drop(['SalePrice'], axis=1)
#相関の散布図から使う説明変数を決め、その説明変数だけをデータから抜き出す

select_train = df.filter(['OverallQual', 'GrLivArea', 'TotRmsAbvGrd']) 

print(select_train.shape)

select_train.head()
#相関の散布図から使う説明変数を決め、その説明変数だけをデータから抜き出す（test ver）

select_test = df_test.filter(['OverallQual', 'GrLivArea', 'TotRmsAbvGrd']) 

print(select_test.shape)

select_test.head()
#欠損値の確認

select_train.isnull().sum()
#欠損値の確認

select_test.isnull().sum()
#訓練データ,テストデータ,教師データを標準化

X_train = (select_train - select_train.mean()) / select_train.std()

X_test = (select_test - select_test.mean()) / select_test.std()
#訓練データの確認

X_train
#テストデータの確認

X_test
#X；選んで処理した説明千数、y；正規分布に近づけたSalePriceをセット

X = X_train[["OverallQual", "GrLivArea", "TotRmsAbvGrd"]].values

y = y_train



# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression()



# fit関数で学習開始

slr.fit(X,y)



print('傾き：{0}'.format(slr.coef_))

a1, a2, a3 = slr.coef_



print('y切片: {0}'.format(slr.intercept_))

b = slr.intercept_
X_ts = X_test[["OverallQual", "GrLivArea", "TotRmsAbvGrd"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = np.exp(slr.predict(X_test))
y_test_pred
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

df_test["SalePrice"] = y_test_pred
# Id, SalePriceの2列だけ表示

df_test[["Id","SalePrice"]].head()
df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)