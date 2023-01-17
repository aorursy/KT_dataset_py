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
 #ライブラリのインポート
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
%matplotlib inline
#データの読み込み
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #テストデータ
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #訓練データ
alldata = pd.concat([train,test],axis=0).reset_index(drop=True)
print("Train data shape:",train.shape)
print("Test data shape:",test.shape)
train.head()
test.head()
train.columns
#SalePriceのデータ数・平均値・標準偏差・最小値・最大値・４分位数を求める
train["SalePrice"].describe()
plt.figure(figsize=(15,8))
sns.boxplot(train.YearBuilt, train.SalePrice)
plt.figure(figsize=(12,6))
plt.scatter(x=train.GrLivArea, y=train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
 # トレーニングデータの欠損値数
train_null = train.isnull().sum()
train_null[train_null>0].sort_values(ascending=False)
#欠損を含むカラムのデータ型を確認
na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist() #欠損を含むカラムをリスト化
alldata[na_col_list].dtypes.sort_values() #データ型

 # !相関係数を算出
corrmat = train.corr()
corrmat
# 算出した相関係数を相関が高い順に上位10個のデータを表示

# ヒートマップに表示させるカラムの数
n = 10

# SalesPriceとの相関が大きい上位10個のカラム名を取得
cols = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index

# SalesPriceとの相関が大きい上位10個のカラムを対象に相関を算出
# .T(Trancepose[転置行列])を行う理由は、corrcoefで相関を算出する際に、各カラムの値を行毎にまとめなければならない為
cm = np.corrcoef(train[cols].values.T)

# ヒートマップのフォントサイズを指定
sns.set(font_scale=1.25)

# 算出した相関データをヒートマップで表示
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# corrmat.nlargest(n, 'SalePrice') : SalePriceの値が大きい順にkの行数だけ抽出した行と全列の行列データ
# corrmat.nlargest(n, 'SalePrice')['SalePrice'] : SalePriceの値が大きい順にnの行数だけ抽出した抽出した行とSalePrice列だけ抽出したデータ
# corrmat.nlargest(n, 'SalePrice')['SalePrice'].index : 行の項目を抽出。データの中身はIndex(['SalePrice', 'OverallQual', ... , 'YearBuilt'], dtype='object')
# cols.values : カラムの値(SalesPrice, OverallQual, ...)
#"SalePrice"と各データの散布図を作成
plt.figure(figsize=(16,12))

#"OverallQual"の散布図
plt.subplot(3, 3, 1)
plt.scatter(train["OverallQual"],train["SalePrice"])

#"GrLivArea"の散布図
plt.subplot(3, 3, 2)
plt.scatter(train["GrLivArea"],train["SalePrice"])

#"GarageCars"の散布図
plt.subplot(3, 3, 3)
plt.scatter(train["GarageCars"],train["SalePrice"])

#"GarageArea"の散布図
plt.subplot(3, 3, 4)
plt.scatter(train["GarageArea"],train["SalePrice"])

#"TotalBsmtSF"の散布図
plt.subplot(3, 3, 5)
plt.scatter(train["TotalBsmtSF"],train["SalePrice"])

#"1stFlrSF"の散布図
plt.subplot(3, 3, 6)
plt.scatter(train["1stFlrSF"],train["SalePrice"])

#"FullBath"の散布図
plt.subplot(3, 3, 7)
plt.scatter(train["FullBath"],train["SalePrice"])

#"TotRmsAbvGrd"の散布図
plt.subplot(3, 3, 8)
plt.scatter(train["TotRmsAbvGrd"],train["SalePrice"])

#"YearBuilt"の散布図
plt.subplot(3, 3, 9)
plt.scatter(train["YearBuilt"],train["SalePrice"])

plt.show()
#GrLivAreaの外れ値を検索
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
# 判明したデータのIdの値を指定して削除
train = train.drop(index = train[train['Id'] == 1299].index)
train = train.drop(index = train[train['Id'] == 524].index)
#"SalePrice"と各データの散布図を作成
plt.figure(figsize=(16,12))

#"OverallQual"の散布図
plt.subplot(3, 3, 1)
plt.scatter(train["OverallQual"],train["SalePrice"])

#"GrLivArea"の散布図
plt.subplot(3, 3, 2)
plt.scatter(train["GrLivArea"],train["SalePrice"])

#"GarageCars"の散布図
plt.subplot(3, 3, 3)
plt.scatter(train["GarageCars"],train["SalePrice"])

#"GarageArea"の散布図
plt.subplot(3, 3, 4)
plt.scatter(train["GarageArea"],train["SalePrice"])

#"TotalBsmtSF"の散布図
plt.subplot(3, 3, 5)
plt.scatter(train["TotalBsmtSF"],train["SalePrice"])

#"1stFlrSF"の散布図
plt.subplot(3, 3, 6)
plt.scatter(train["1stFlrSF"],train["SalePrice"])

#"FullBath"の散布図
plt.subplot(3, 3, 7)
plt.scatter(train["FullBath"],train["SalePrice"])

#"TotRmsAbvGrd"の散布図
plt.subplot(3, 3, 8)
plt.scatter(train["TotRmsAbvGrd"],train["SalePrice"])

#"YearBuilt"の散布図
plt.subplot(3, 3, 9)
plt.scatter(train["YearBuilt"],train["SalePrice"])

plt.show()
# XにGrLivArea、yにSalePriceをセット
X = train[["GrLivArea"]].values
y = train["SalePrice"].values

# アルゴリズムに線形回帰(Linear Regression)を採用
slr = LinearRegression()

# fit関数で学習開始
slr.fit(X,y)

# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力
# 偏回帰係数はscikit-learnのcoefで取得
print('傾き：{0}'.format(slr.coef_[0]))

# y切片(直線とy軸との交点)を出力
# 余談：x切片もあり、それは直線とx軸との交点を指す
print('y切片: {0}'.format(slr.intercept_))
# 散布図を描画
plt.scatter(X,y)

# 折れ線グラフを描画
plt.plot(X,slr.predict(X),color='red')

# 表示
plt.show()
# テストデータの GrLivArea の値をセット
X_test = test[["GrLivArea"]].values

# 学習済みのモデルから予測した結果をセット
y_test_pred = slr.predict(X_test)
 # df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット
test["SalePrice"] = y_test_pred
# Id, SalePriceの2列だけ表示
test[["Id","SalePrice"]].head()
 # Id, SalePriceの2列だけのファイルに変換
test[["Id","SalePrice"]].to_csv("submission.csv",index=False)
