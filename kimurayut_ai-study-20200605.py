from sklearn import datasets

# ボストンの住宅価格データセットの読み込み

boston = datasets.load_boston()
# データセットの情報を調べる

# bostonデータセットの詳細な説明

print(boston.DESCR)
#グラフ描画、計算用のライブラリ

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline 
#特徴量

X = boston.data

#正解ラベル

y = boston.target
#DataFrame化することによってデータの描画や編集がし易い形にする

df_X = pd.DataFrame(X)

df_y = pd.DataFrame(y)

#特徴量のカラム

df_X.columns = boston.feature_names

#正解ラベルのカラム

df_y.columns = ["target"]
#特徴量データの先頭5件を表示する

df_X.head()
#正解データの先頭5件を表示する

df_y.head()
#特徴量と正解データを結合する

df = pd.concat([df_X, df_y], axis = 1)

df.head()
#相関係数の計算

#平均値 → 偏差 → 分散 → 標準偏差 → 共分散の順に計算した結果を返してくれる

#1に近いほど、強い正の相関があり、−1に近いほど、強い負の相関があります。

corr = df.corr()

corr
import seaborn as sns

sns.heatmap(corr,

            vmin=-1.0,

            vmax=1.0,

            center=0,

            annot=True, # True:格子の中に値を表示

            fmt='.1f',

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values

           )

plt.show()
lstat = df_X.loc[:,"LSTAT"]

lstat
# lstatのデータ型はpandas.core.series.Seriesと特殊な形になるので、

# このままだと機械学習ができません。

# そのためlstatをnumpyの配列に変換し、二次元ベクトルにする

print("変換前のデータ型は", type(lstat))

print("変換前の配列の大きさは", lstat.shape)

lstat = lstat.values

lstat = lstat.reshape(-1,1)

print("変換後のデータ型は", type(lstat))

print("変換後の配列の大きさは", lstat.shape)
plt.scatter(lstat, y)

plt.show()

#x軸がLSTAT: 給与の低い職業に従事する人口の割合 (%)

#y軸が住宅価格
#学習データと評価データで分割

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(lstat, y, test_size=0.3, random_state=0)
#線形回帰モデルの実装

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
#学習

lr.fit(X_train, y_train)
# 予測結果をグラフ描画

# 赤い線が予測した結果

plt.scatter(lstat, y)                       

plt.plot(X_test, lr.predict(X_test), color='red') 

plt.title('boston_housing')     

plt.xlabel('LSTAT')               

plt.ylabel('target')                 

plt.show()
y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



# 性能評価

# 二乗平均平方根誤差(Root Mean Squared Error/RMSE)および、

# 決定係数(Coefficient of Determination)という評価指標を使います

from math import sqrt

from sklearn.metrics import mean_squared_error

# 二乗平均平方根誤差(RMSE)を算出

print('RMSE Test :' + str((sqrt(mean_squared_error(y_test, y_test_pred)))))

# 学習用、検証用データに関してR^2を出力 (回帰モデルの場合score()を使うことで決定係数が得られます。)

print('R^2 Train : %.3f, Test : %.3f' % (lr.score(X_train, y_train), lr.score(X_test, y_test)))
x = np.arange(15,16)

x = x.reshape(-1,1)

print("LSTAT(給与の低い職業に従事する人口の割合 (%))")

print(x)

print("予測値")

print(lr.predict(x))
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)



X_train = X_train[["RM","LSTAT"]] #部屋数と給与が低い人口割合での重回帰

X_test = X_test[["RM","LSTAT"]]
from sklearn.linear_model import LinearRegression

lrt = LinearRegression()

lrt.fit(X_train, y_train)
y_train_pred = lrt.predict(X_train)

y_test_pred = lrt.predict(X_test)



# 性能評価

# 二乗平均平方根誤差(Root Mean Squared Error/RMSE)および、

# 決定係数(Coefficient of Determination)という評価指標を使います

from math import sqrt

from sklearn.metrics import mean_squared_error

# 二乗平均平方根誤差(RMSE)を算出

print('RMSE Test :' + str((sqrt(mean_squared_error(y_test, y_test_pred)))))
