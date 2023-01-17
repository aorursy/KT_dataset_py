# ライブラリのインポート

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression



# Jupyter Notebookの中でインライン表示する場合の設定（これが無いと別ウィンドウでグラフが開く）



%matplotlib inline 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# データの読み込み

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df
# 先頭5行の確認

df.head()
# カラムの一覧表示

df.columns
# 基本統計量の表示

df.SalePrice.describe()
# ヒストグラムの表示

sns.distplot(df.SalePrice)
# 相関係数を算出

corrmat = df.corr()

corrmat
# 算出した相関係数を相関が高い順に上位10個のデータを表示



# ヒートマップに表示させるカラムの数

k = 10



# SalesPriceとの相関が大きい上位10個のカラム名を取得

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index



# SalesPriceとの相関が大きい上位10個のカラムを対象に相関を算出

# .T(Trancepose[転置行列])を行う理由は、corrcoefで相関を算出する際に、各カラムの値を行毎にまとめなければならない為

cm = np.corrcoef(df[cols].values.T)



# ヒートマップのフォントサイズを指定

sns.set(font_scale=1.25)



# 算出した相関データをヒートマップで表示

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
k = 38

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice']

cols
# 散布図の表示

# sns.set()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']

# sns.pairplot(df[cols], size = 2.5)

# plt.show()
# 数値の大きい上位2位のデータを表示

df.sort_values(by = 'GrLivArea', ascending = False)[:2]
# No.21で判明したデータのIdの値を指定して削除

df = df.drop(index = df[df['Id'] == 1299].index)

df = df.drop(index = df[df['Id'] == 524].index)
# 散布図の表示

# sns.set()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']

# sns.pairplot(df[cols], size = 2.5)

# plt.show()
df.sort_values(by = 'GarageArea', ascending = False)[:3]
df = df.drop(index = df[df['Id'] == 582].index)

df = df.drop(index = df[df['Id'] == 1191].index)

df = df.drop(index = df[df['Id'] == 1062].index)
# 散布図の表示

# sns.set()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']

# sns.pairplot(df[cols], size = 2.5)

# plt.show()
df.sort_values(by = 'TotalBsmtSF', ascending = False)[:3]
df = df.drop(index = df[df['Id'] == 333].index)

df = df.drop(index = df[df['Id'] == 497].index)

df = df.drop(index = df[df['Id'] == 441].index)
# 散布図の表示

# sns.set()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']

# sns.pairplot(df[cols], size = 2.5)

# plt.show()
df.sort_values(by = '1stFlrSF', ascending = False)[:1]
df = df.drop(index = df[df['Id'] == 1025].index)
# 散布図の表示

# sns.set()

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']

# sns.pairplot(df[cols], size = 2.5)

# plt.show()
sns.distplot(np.log(df['SalePrice']))
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Lasso



# XにGrLivArea、yにSalePriceをセット

X = df[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']]

y = np.log(df['SalePrice'])
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error



scaler = StandardScaler()  #スケーリング

param_grid = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015] #パラメータグリッド

cnt = 0

for alpha in param_grid:

    ls = Lasso(alpha=alpha) #Lasso回帰モデル

    pipeline = make_pipeline(scaler, ls) #パイプライン生成

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    elif best_score > test_rmse:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    else:

        pass

    cnt = cnt + 1

    

print('alpha : ' + str(best_param))

print('test score is : ' +str(best_score))
# アルゴリズムに線形回帰(Linear Regression)を採用

slr = Lasso(alpha=0.0006)



# fit関数で学習開始

slr.fit(X,y)



# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力

# 偏回帰係数はscikit-learnのcoefで取得

print('傾き：{0}'.format(slr.coef_))





# y切片(直線とy軸との交点)を出力

# 余談：x切片もあり、それは直線とx軸との交点を指す

print('y切片: {0}'.format(slr.intercept_))

b = slr.intercept_
# テストデータの読込

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

df_test['GarageArea'].fillna(478, inplace=True)

df_test['TotalBsmtSF'].fillna(990, inplace=True)
# テストデータの内容確認(追加したSalePriceが消えている事)

df_test.head()
# テストデータの OverallQual と GrLivArea の値をセット

X_test = df_test[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']].values

X_test[np.isnan(X_test).any(axis=1), :] # 2121, 2577

    

# print(df.median()) # 478, 990



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)
# 学習済みのモデルから予測した結果を出力

y_test_pred
y_test_pred = np.exp(y_test_pred)
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

df_test["SalePrice"] = y_test_pred
# Id, SalePriceの2列だけ表示

df_test[["Id","SalePrice"]].head()
# Id, SalePriceの2列だけのファイルに変換

df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)