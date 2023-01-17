%matplotlib inline

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression, SGDClassifier

from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数

from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画

import math

from sklearn.linear_model import Ridge,Lasso,ElasticNet #正則化項付き最小二乗法を行うためのライブラリ

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from IPython.display import display

from sklearn.preprocessing import StandardScaler, MinMaxScaler

%matplotlib inline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.metrics import mean_absolute_error # 回帰問題における性能評価に関する関数

import matplotlib.pyplot as plt
df = pd.read_excel("../input/measurements2.xlsx")
#specials,refill liters,refill gasのカラムを削除

df = df.drop(['specials','refill liters','refill gas'],axis=1)
#ダミー変数　E10,SP98を変換

df = pd.get_dummies(df)

#NaNを0に置換

df = df.fillna(0)
#AC,gas_type_SP98のカラムを削除

df = df.drop(['AC','gas_type_SP98'],axis=1)

#E10を1,SP98を0として扱う
#データの表示

display(df.loc[:, ['consume','distance','speed','temp_inside','temp_outside','rain', 'sun','gas_type_E10']].head())

df.describe()
# 散布図行列

pd.plotting.scatter_matrix(df,figsize=(20,20))

plt.show()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 12))

df.plot(kind='scatter', ax=axes[0, 0], legend=False,x=u'distance', y=u'consume', c="gas_type_E10", cmap='winter',sharex=False)

df.plot(kind='scatter', ax=axes[0, 1], legend=False,x=u'speed', y=u'consume', c="gas_type_E10", cmap='winter',sharex=False)

df.plot(kind='scatter', ax=axes[1, 0], legend=False,x=u'temp_inside', y=u'consume', c="gas_type_E10", cmap='winter',sharex=False)

df.plot(kind='scatter', ax=axes[1, 1], legend=False,x=u'temp_outside', y=u'consume', c="gas_type_E10", cmap='winter',sharex=False)

df.plot(kind='scatter', ax=axes[2, 0], legend=False,x=u'rain', y=u'consume', c="gas_type_E10", cmap='winter',sharex=False)

df.plot(kind='scatter', ax=axes[2, 1], legend=False,x=u'sun', y=u'consume', c="gas_type_E10", cmap='winter',sharex=False)

axes[0, 0].set_xlabel("Distance[km]")

axes[0, 1].set_xlabel("Speed [km/h]")

axes[1, 0].set_xlabel("Temp_inside [℃]")

axes[1, 1].set_xlabel("Temp_outside [℃]")

axes[2, 0].set_xlabel("Rain")

axes[2, 1].set_xlabel("Sun")

axes[0, 0].set_ylabel("Consume [L/100km]")

axes[0, 1].set_ylabel("Consume [L/100km]")

axes[1, 0].set_ylabel("Consume [L/100km]")

axes[1, 1].set_ylabel("Consume [L/100km]")

axes[2, 0].set_ylabel("Consume [L/100km]")

axes[2, 1].set_ylabel("Consume [L/100km]")
# 前処理以前のデータで線形回帰を実行した場合のMSE,MAE,RMSEを求める

# 係数を求める

y = df["consume"].values

X = df[["distance", "speed", "temp_inside", "temp_outside","gas_type_E10","rain","sun"]].values



regr = LinearRegression(fit_intercept=True)

regr.fit(X, y)



# 重みを取り出す

w0 = regr.intercept_

w1 = regr.coef_[0]

w2 = regr.coef_[1]

w3 = regr.coef_[2]

w4 = regr.coef_[3]

w5 = regr.coef_[4]

w6 = regr.coef_[5]

w7 = regr.coef_[6]
x1 = df['distance'].values

x2 = df['speed'].values

x3 = df['temp_inside'].values

x4 = df['temp_outside'].values

x5 = df['gas_type_E10'].values

x6 = df['rain'].values

x7 = df['sun'].values



# 重みと二乗誤差の確認 

y_est = w0 + w1*x1 + w2*x2 +w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7

squared_error = 0.5 * np.sum((y - y_est) ** 2)

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}, w7 = {:.3f}'.format(w0, w1, w2,w3,w4,w5,w6,w7))

print('二乗誤差 = {:.3f}'.format(squared_error))
# 線形回帰

regr = LinearRegression(fit_intercept=True)

regr.fit(X, y)
# 値を予測

y_pred = regr.predict(X)



# MSEを計算

mse = mean_squared_error(y, y_pred) 

print("MSE = %s"%round(mse,3) )  



# MAEを計算

mae = mean_absolute_error(y, y_pred) 

print("MAE = %s"%round(mae,3) )



# RMSEを計算

rmse = np.sqrt(mse)

print("RMSE = %s"%round(rmse, 3) )
from sklearn.model_selection import train_test_split

X, y = df.iloc[:, [0,2,3,4,5,6,7]].values, df["consume"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#正規化の実行（学習用データ）

normsc = MinMaxScaler()

X_train_norm = normsc.fit_transform(X_train)
# 学習用データを使って線形回帰モデルを学習

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train_norm, y_train)



# 学習用データに対する予測を実行

y_pred_train = regr.predict(X_train_norm)



# 学習データに対するMAEを計算（訓練誤差の評価）

mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
#正規化の実行（テストデータ）

X_test_norm = normsc.transform(X_test)
# テストデータに対する予測を実行

y_pred_test = regr.predict(X_test_norm)



# テストデータに対するMAEを計算（汎化誤差の評価）

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
#標準化の実行（学習用データ）

stdsc = StandardScaler()

X_train_stand = stdsc.fit_transform(X_train)
# 学習用データを使って線形回帰モデルを学習

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train_stand, y_train)



# 学習用データに対する予測を実行

y_pred_train = regr.predict(X_train_stand)



# 学習データに対するMAEを計算（訓練誤差の評価）

mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
#標準化の実行（テストデータ）

X_test_stand = stdsc.transform(X_test)
# テストデータに対する予測を実行

y_pred_test = regr.predict(X_test_stand)



# テストデータに対するMAEを計算（汎化誤差の評価）

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
#標準化された学習用データに対して相関係数を求める

X_train_stand_pd = X_train_stand

df_X_train_stand = pd.DataFrame(X_train_stand_pd)

df_X_train_stand.corr()
# 相関係数をヒートマップにして可視化

sns.heatmap(df_X_train_stand.corr())

plt.show()
#学習用データを白色化（標準化された学習用データを無相関化 ）

cov = np.cov(X_train_stand, rowvar=0) # 分散・共分散を求める

_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

X_train_stand_decorr = np.dot(S.T, X_train_stand.T).T #データを無相関化
#白色化された学習用データに対して相関係数を求める

X_train_stand_decorr_pd = X_train_stand_decorr

df_X_train_stand_decorr = pd.DataFrame(X_train_stand_decorr_pd)

df_X_train_stand_decorr.corr()
# 相関係数をヒートマップにして可視化

sns.heatmap(df_X_train_stand_decorr.corr())

plt.show()
# 学習用データを使って線形回帰モデルを学習

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train_stand, y_train)

# 学習用データに対する予測を実行

y_pred_train = regr.predict(X_train_stand_decorr)

# 学習データに対するMAEを計算（訓練誤差の評価）

mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
#テストデータを白色化（標準化されたテストデータを無相関化 ）

cov = np.cov(X_test_stand, rowvar=0) # 分散・共分散を求める

_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

X_test_stand_decorr = np.dot(S.T, X_test_stand.T).T #データを無相関化
# テストデータに対する予測を実行

y_pred_test = regr.predict(X_test_stand_decorr)

# テストデータに対するMAEを計算（汎化誤差の評価）

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
# グラフに重ねて表示する

plt.plot(y_train,(y_pred_train - y_train), 'o', label='training data')

plt.plot(y_test, (y_pred_test - y_test), '^', label='test data')

plt.ylabel("Y_pred - Y")

plt.xlabel("Consume [L/100km]")

plt.grid(which='major',color='black',linestyle=':')

plt.grid(which='minor',color='black',linestyle=':')

#plt.plot(X_train[:,0], y_pred_train, label='regression')

plt.legend(loc='best')

plt.show()
n_split = 10 # グループ数を設定



cross_valid_mae = 0

split_num = 1



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #標準化の実行（学習用データ）

    stdsc = StandardScaler()

    X_train_stand = stdsc.fit_transform(X_train)

    #学習用データを白色化（標準化された学習用データを無相関化 ）

    cov = np.cov(X_train_stand, rowvar=0) # 分散・共分散を求める

    _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

    X_train_stand_decorr = np.dot(S.T, X_train_stand.T).T #データを無相関化

    #標準化の実行（テストデータ）

    X_test_stand = stdsc.transform(X_test)

    #テストデータを白色化（標準化されたテストデータを無相関化 ）

    cov = np.cov(X_test_stand, rowvar=0) # 分散・共分散を求める

    _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

    X_test_stand_decorr = np.dot(S.T, X_test_stand.T).T #データを無相関化

   

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_train_stand_decorr, y_train)



    # テストデータに対する予測を実行

    regr.fit(X_test_stand_decorr, y_test)

    y_pred_test = regr.predict(X_test_stand_decorr)

    

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
#上記を正則化する

#Ridgeの場合



cross_valid_mae = 0

split_num = 1



alphas = [0.0, 1e-8, 1e-5, 1e-1]

for alpha in zip(alphas):

    print('alpha =',alpha)

    # テスト役を交代させながら学習と評価を繰り返す

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

        X_train, y_train = X[train_idx], y[train_idx] #学習用データ

        X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

        #標準化の実行（学習用データ）

        stdsc = StandardScaler()

        X_train_stand = stdsc.fit_transform(X_train)

        #学習用データを白色化（標準化された学習用データを無相関化 ）

        cov = np.cov(X_train_stand, rowvar=0) # 分散・共分散を求める

        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

        X_train_stand_decorr = np.dot(S.T, X_train_stand.T).T #データを無相関化

        #標準化の実行（テストデータ）

        X_test_stand = stdsc.transform(X_test)

        #テストデータを白色化（標準化されたテストデータを無相関化 ）

        cov = np.cov(X_test_stand, rowvar=0) # 分散・共分散を求める

        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

        X_test_stand_decorr = np.dot(S.T, X_test_stand.T).T #データを無相関化

   

        # 学習用データを使って線形回帰モデルを学習

        model_Ridge = Ridge(alpha = alpha)

        model_Ridge.fit(X_train_stand_decorr, y_train)



        # テストデータに対する予測を実行

        model_Ridge.fit(X_test_stand_decorr, y_test)

        y_pred_test = model_Ridge.predict(X_test_stand_decorr)

    

        # テストデータに対するMAEを計算

        mae = mean_absolute_error(y_test, y_pred_test)

        print("Fold %s"%split_num)

        print("MAE = %s"%round(mae, 3))

        print()

    

        cross_valid_mae += mae #後で平均を取るためにMAEを加算

        split_num += 1



    # MAEの平均値を最終的な汎化誤差値とする

    final_mae = cross_valid_mae / n_split

    print("Cross Validation MAE = %s"%round(final_mae, 3))
print(model_Ridge.intercept_) 

print(model_Ridge.coef_) 
#上記を正則化する

#Lassoの場合



cross_valid_mae = 0

split_num = 1



alphas = [1e-20, 1e-3, 1e-2, 1e-1]#alpha(数式ではλ)の値を4つ指定する

for alpha in zip(alphas):

    print('alpha =',alpha)

    # テスト役を交代させながら学習と評価を繰り返す

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

        X_train, y_train = X[train_idx], y[train_idx] #学習用データ

        X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

        #標準化の実行（学習用データ）

        stdsc = StandardScaler()

        X_train_stand = stdsc.fit_transform(X_train)

        #学習用データを白色化（標準化された学習用データを無相関化 ）

        cov = np.cov(X_train_stand, rowvar=0) # 分散・共分散を求める

        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

        X_train_stand_decorr = np.dot(S.T, X_train_stand.T).T #データを無相関化

        #標準化の実行（テストデータ）

        X_test_stand = stdsc.transform(X_test)

        #テストデータを白色化（標準化されたテストデータを無相関化 ）

        cov = np.cov(X_test_stand, rowvar=0) # 分散・共分散を求める

        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

        X_test_stand_decorr = np.dot(S.T, X_test_stand.T).T #データを無相関化

   

        # 学習用データを使って線形回帰モデルを学習

        model_Lasso = Lasso(alpha=alpha, max_iter=1e5)

        model_Lasso.fit(X_train_stand_decorr, y_train)



        # テストデータに対する予測を実行

        model_Lasso.fit(X_test_stand_decorr, y_test)

        y_pred_test = model_Lasso.predict(X_test_stand_decorr)

    

        # テストデータに対するMAEを計算

        mae = mean_absolute_error(y_test, y_pred_test)

        print("Fold %s"%split_num)

        print("MAE = %s"%round(mae, 3))

        print()

    

        cross_valid_mae += mae #後で平均を取るためにMAEを加算

        split_num += 1



    # MAEの平均値を最終的な汎化誤差値とする

    final_mae = cross_valid_mae / n_split

    print("Cross Validation MAE = %s"%round(final_mae, 3))
print(model_Lasso.intercept_) 

print(model_Lasso.coef_) 
#上記を正則化する

#ElasticNet の場合



cross_valid_mae = 0

split_num = 1



alpha = 1e-4 #正則化全体の強さを決定する

l1_ratios = [0, 0.1, 0.5, 1.0] #L1正則化の強さを4つ指定する（L2正則化の強さは1 - l1_ratioで自動的に設定される）

for l1_ratio in zip(l1_ratios):

    print('alpha =',alpha)

    # テスト役を交代させながら学習と評価を繰り返す

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

        X_train, y_train = X[train_idx], y[train_idx] #学習用データ

        X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

        #標準化の実行（学習用データ）

        stdsc = StandardScaler()

        X_train_stand = stdsc.fit_transform(X_train)

        #学習用データを白色化（標準化された学習用データを無相関化 ）

        cov = np.cov(X_train_stand, rowvar=0) # 分散・共分散を求める

        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

        X_train_stand_decorr = np.dot(S.T, X_train_stand.T).T #データを無相関化

        #標準化の実行（テストデータ）

        X_test_stand = stdsc.transform(X_test)

        #テストデータを白色化（標準化されたテストデータを無相関化 ）

        cov = np.cov(X_test_stand, rowvar=0) # 分散・共分散を求める

        _, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて

        X_test_stand_decorr = np.dot(S.T, X_test_stand.T).T #データを無相関化

   

        # 学習用データを使って線形回帰モデルを学習

        model_ElasticNet  = make_pipeline(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1e5))

        model_ElasticNet.fit(X_train_stand_decorr, y_train)



        # テストデータに対する予測を実行

        model_ElasticNet .fit(X_test_stand_decorr, y_test)

        y_pred_test = model_ElasticNet.predict(X_test_stand_decorr)

    

        # テストデータに対するMAEを計算

        mae = mean_absolute_error(y_test, y_pred_test)

        print("Fold %s"%split_num)

        print("MAE = %s"%round(mae, 3))

        print()

    

        cross_valid_mae += mae #後で平均を取るためにMAEを加算

        split_num += 1



    # MAEの平均値を最終的な汎化誤差値とする

    final_mae = cross_valid_mae / n_split

    print("Cross Validation MAE = %s"%round(final_mae, 3))