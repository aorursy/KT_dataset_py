# No.1
# ライブラリのインポート
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Jupyter Notebookの中でインライン表示する場合の設定（これが無いと別ウィンドウでグラフが開く）
%matplotlib inline 

# No.2
# データの読み込み
df = pd.read_csv("../input/train.csv")
# No.3
# 読み込んだデータの確認
df
# No.4
# 先頭5行の確認
df.head()
# No.5
# XにOverallQual、yにSalePriceをセット
X = df[["OverallQual"]].values
y = df["SalePrice"].values

# アルゴリズムに線形回帰(Linear Regression)を採用
slr = LinearRegression()

# fit関数でモデル作成
slr.fit(X,y)

# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力
# 偏回帰係数はscikit-learnのcoefで取得
print('傾き：{0}'.format(slr.coef_[0]))

# y切片(直線とy軸との交点)を出力
# 余談：x切片もあり、それは直線とx軸との交点を指す
print('y切片: {0}'.format(slr.intercept_))
# No.6
# 散布図を描画
plt.scatter(X,y)

# 折れ線グラフを描画
plt.plot(X,slr.predict(X),color='red')

# 表示
plt.show()
# No.7
# テストデータの読込
df_test = pd.read_csv("../input/test.csv")
# No.8
# テストデータの内容確認(評価用のデータなので、SalePriceはない)
df_test.head()
# No.9
# テストデータの OverallQual の値をセット
X_test = df_test[["OverallQual"]].values

# 学習済みのモデルから予測した結果をセット
y_test_pred = slr.predict(X_test)
# No.10
# 学習済みのモデルから予測した結果を出力
y_test_pred
# No.11
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット
df_test["SalePrice"] = y_test_pred
# No.12
# df_testの先頭5行を確認
df_test.head()
# No.13
# Id, SalePriceの2列だけ表示
df_test[["Id","SalePrice"]].head()
# No.14
# Id, SalePriceの2列だけのファイルに変換
df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)
# No.15
# カラムの一覧表示
df.columns
# No.16
# 基本統計量の表示
df.SalePrice.describe()
# No.17
# ヒストグラムの表示
sns.distplot(df.SalePrice)
# No.18
# 相関係数を算出
corrmat = df.corr()
corrmat
# No.19
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
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# corrmat.nlargest(k, 'SalePrice') : SalePriceの値が大きい順にkの行数だけ抽出した行と全列の行列データ
# corrmat.nlargest(k, 'SalePrice')['SalePrice'] : SalePriceの値が大きい順にkの行数だけ抽出した抽出した行とSalePrice列だけ抽出したデータ
# corrmat.nlargest(k, 'SalePrice')['SalePrice'].index : 行の項目を抽出。データの中身はIndex(['SalePrice', 'OverallQual', ... , 'YearBuilt'], dtype='object')
# cols.values : カラムの値(SalesPrice, OverallQual, ...)
# No.20
# 散布図の表示
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea']
sns.pairplot(df[cols], size = 2.5)
plt.show()
# No.21
# 数値の大きい上位2位のデータを表示
df.sort_values(by = 'GrLivArea', ascending = False)[:2]
#ascending=False は'GrLivArea'の高いデータから順に並べ替えられる
# No.22
# No.21で判明したデータのIdの値を指定して削除
df = df.drop(index = df[df['Id'] == 1299].index)
df = df.drop(index = df[df['Id'] == 524].index)
# No.23
# 散布図の表示
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea']
sns.pairplot(df[cols], size = 2.5)
plt.show()
# No.24
# XにGrLivArea、yにSalePriceをセット
X = df[["GrLivArea"]].values
y = df["SalePrice"].values

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
# No.25
# 散布図を描画
plt.scatter(X,y)

# 折れ線グラフを描画
plt.plot(X,slr.predict(X),color='red')

# 表示
plt.show()
# No.26
# テストデータの読込
df_test = pd.read_csv("../input/test.csv")
# No.27
# テストデータの内容確認(最初にモデルを作成した際に追加したSalePriceが消えている事)
df_test.head()
# No.28
# テストデータの GrLivArea の値をセット
X_test = df_test[["GrLivArea"]].values

# 学習済みのモデルから予測した結果をセット
y_test_pred = slr.predict(X_test)
# No.29
# 学習済みのモデルから予測した結果を出力
y_test_pred
# No.30
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット
df_test["SalePrice"] = y_test_pred
# No.31
# Id, SalePriceの2列だけ表示
df_test[["Id","SalePrice"]].head()
# No.32
# Id, SalePriceの2列だけのファイルに変換
df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)
# No.33
# XにGrLivArea、yにSalePriceをセット
X = df[["OverallQual", "GrLivArea"]].values
y = df["SalePrice"].values

# アルゴリズムに線形回帰(Linear Regression)を採用
slr = LinearRegression()

# fit関数で学習開始
slr.fit(X,y)

# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力
# 偏回帰係数はscikit-learnのcoefで取得
print('傾き：{0}'.format(slr.coef_))
a1, a2 = slr.coef_

# y切片(直線とy軸との交点)を出力
# 余談：x切片もあり、それは直線とx軸との交点を指す
print('y切片: {0}'.format(slr.intercept_))
b = slr.intercept_
# No.34
# 3D描画（散布図の描画）
x, y, z = np.array(df["OverallQual"]), np.array(df["GrLivArea"]), np.array(df["SalePrice"].values)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(np.ravel(x), np.ravel(y), np.ravel(z))

# 3D描画（回帰平面の描画）
# np.arange(0, 10, 2)は# 初項0,公差2で終点が10の等差数列(array([ 2,  4,  6,  8, 10]))
X, Y = np.meshgrid(np.arange(0, 12, 2), np.arange(0, 6000, 1000))
Z = a1 * X + a2 * Y + b
ax.plot_surface(X, Y, Z, alpha = 0.5, color = "red") #alphaで透明度を指定
ax.set_xlabel("OverallQual")
ax.set_ylabel("GrLivArea")
ax.set_zlabel("SalePrice")

plt.show()
# No.35
# テストデータの読込
df_test = pd.read_csv("../input/test.csv")
# No.36
# テストデータの内容確認(追加したSalePriceが消えている事)
df_test.head()
# No.37
# テストデータの OverallQual と GrLivArea の値をセット
X_test = df_test[["OverallQual", "GrLivArea"]].values

# 学習済みのモデルから予測した結果をセット
y_test_pred = slr.predict(X_test)
# No.38
# 学習済みのモデルから予測した結果を出力
y_test_pred
# No.39
# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット
df_test["SalePrice"] = y_test_pred
# No.40
# Id, SalePriceの2列だけ表示
df_test[["Id","SalePrice"]].head()
# No.41
# Id, SalePriceの2列だけのファイルに変換
df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)