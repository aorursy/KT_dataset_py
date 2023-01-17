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

df
# カラムの一覧表示

df.columns
# 基本統計量の表示

df.SalePrice.describe()
# ヒストグラムの表示

sns.distplot(df.SalePrice)
# 相関係数を算出

corrmat = df.corr()

corrmat
# 算出した相関係数を相関が高い順に上位5個のデータを表示



# ヒートマップに表示させるカラムの数

k = 5



# SalesPriceとの相関が大きい上位5個のカラム名を取得

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index



# SalesPriceとの相関が大きい上位5個のカラムを対象に相関を算出

cm = np.corrcoef(df[cols].values.T)



# ヒートマップのフォントサイズ

sns.set(font_scale=1.5)



# 算出した相関データのヒートマップ

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# 散布図の表示

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(df[cols], size = 2.5)

plt.show()
# 数値の大きい上位2位のデータを表示

df.sort_values(by = 'GrLivArea', ascending = False)[:2]
# 大幅にずれたデータのIdの値を削除

df = df.drop(index = df[df['Id'] == 1299].index)

df = df.drop(index = df[df['Id'] == 524].index)
# 散布図の表示

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(df[cols], size = 2.5)

plt.show()
# XにGrLivArea、yにSalePrice

X = df[["GrLivArea"]].values

y = df["SalePrice"].values



# アルゴリズムに線形回帰

slr = LinearRegression()



# fit関数

slr.fit(X,y)



# 偏回帰係数を出力

# 偏回帰係数はscikit-learnのcoefで取得

print('傾き：{0}'.format(slr.coef_[0]))



# y切片を出力

# 余談：x切片もあり、それは直線とx軸との交点を指す

print('y切片: {0}'.format(slr.intercept_))
# 散布図を描画

plt.scatter(X,y)



# 折れ線グラフを描画

plt.plot(X,slr.predict(X),color='red')



# 表示

plt.show()
# テストデータの読込

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

df
# テストデータの GrLivArea の値をセット

X_test = df_test[["GrLivArea"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)



# 学習済みのモデルから予測した結果を出力

y_test_pred
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

df_test["SalePrice"] = y_test_pred

# Id, SalePriceの2列だけ表示

df_test[["Id","SalePrice"]]
# Id, SalePriceの2列だけのファイルに変換

df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)