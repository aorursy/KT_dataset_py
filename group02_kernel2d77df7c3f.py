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
# データの読み込み

df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df.head(10)
# カラムの一覧表示

df.columns
# 基本統計量

df.SalePrice.describe()
# Sale priceのヒストグラム

sns.distplot(df.SalePrice)
# 相関係数　算出

corrmat = df.corr()

corrmat
# 算出した相関が高い順に上位5個のデータ



# カラム数

k = 5

# SalesPriceとの相関が高い上位5個

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index



# SalesPriceとの相関が高い上位5個のカラム算出

cm = np.corrcoef(df[cols].values.T)



#　フォントサイズ

sns.set(font_scale=1.5)



# ヒートマップ

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# 散布図の表示

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(df[cols], size = 2.5)

plt.show()
# ずれているデータ

df.sort_values(by = 'GrLivArea', ascending = False)[:2]
# 大幅にずれているデータを削除

df = df.drop(index = df[df['Id'] == 1299].index)

df = df.drop(index = df[df['Id'] == 524].index)
# 散布図改訂版

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(df[cols], size = 2.5)

plt.show()
# xにGrLivArea、yにSalePrice

X = df[["GrLivArea"]].values

y = df["SalePrice"].values



# 線形回帰

slr = LinearRegression()



# fit関数

slr.fit(X,y)



# 偏回帰係数

print('傾き：{0}'.format(slr.coef_[0]))



# y切片を出力

print('y切片: {0}'.format(slr.intercept_))
# 散布図

plt.scatter(X,y)



# グラフ

plt.plot(X,slr.predict(X),color='red')



plt.show()
# テストデータの読込

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

df.head(10)
# テストデータのGrLivAreaの値をセット

X_test = df_test[["GrLivArea"]].values



# 予測した結果をセット

y_test_pred = slr.predict(X_test)



# 予測した結果を出力

y_test_pred
# 学習済みのモデルから予測した結果をセット

df_test["SalePrice"] = y_test_pred

# Id, SalePrice表示

df_test[["Id","SalePrice"]].head(10)
df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)