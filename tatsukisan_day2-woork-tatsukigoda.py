%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import sklearn.preprocessing as sp

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge,Lasso,ElasticNet

from sklearn.svm import SVC
#----------データの読み込み--------------

df_car = pd.read_excel("../input/measurements2.xlsx")
#----------形状の確認----------

#df_car.shape
#----------中身の確認----------

#display(df_car.head())
#----------データ型の確認----------

#df_car.dtypes
#----------欠損値の確認----------

#df_car.isnull().sum()
#----------temp_inside_欠損を平均値で補完----------

df_car['temp_inside'] = df_car['temp_inside'].fillna(df_car['temp_inside'].mean())
#----------refill liters_NaNを0で補完----------

df_car['refill liters'] = df_car['refill liters'].fillna(0)
#----------欠損値の再確認----------

#df_car.isnull().sum()
#----------gas_type E10を０、SP98を1　に置き換え----------

df_car['gas_type'] = df_car['gas_type'].replace({'E10': 0, 'SP98': 1})
#----------頭5行を再表示----------

#df_car.head()
#----------specials,refill gas列の削除----------

df_car = df_car.drop(['specials','refill gas'],axis=1)
#----------頭5行を再表示----------

#display(df_car.head())
#----------散布図行列を確認----------

# pd.plotting.scatter_matrix(df_car, figsize=(18,18))

# plt.show()
#----------相関係数を確認----------

#df_car.corr()
#----------相関係数をヒートマップにして可視化----------

# sns.heatmap(df_car.corr())

# plt.show()
#----------係数を求める----------

y = df_car['consume'].values

X = df_car.drop("consume", axis=1).values

regr = LinearRegression()

regr.fit(X, y)
#----------重みを取り出す----------

# w0 = regr.intercept_

# w1 = regr.coef_[0]

# w2 = regr.coef_[1]

# w3 = regr.coef_[2]

# w4 = regr.coef_[3]

# w5 = regr.coef_[4]

# w6 = regr.coef_[5]

# w7 = regr.coef_[6]

# w8 = regr.coef_[7]

# w9 = regr.coef_[8]
# x1 = df_car['distance'].values

# x2 = df_car['speed'].values

# x3 = df_car['temp_inside'].values

# x4 = df_car['temp_outside'].values

# x5 = df_car['gas_type'].values

# x6 = df_car['AC'].values

# x7 = df_car['rain'].values

# x8 = df_car['sun'].values

# x9 = df_car['refill liters'].values

              

# # 重みと二乗誤差の確認

# y_est = w0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7 + w8 * x8 + w9 * x9

# squared_error = 0.5 * np.sum((y - y_est) ** 2)

# print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}, w7 = {:.3f}, w8 = {:.3f}, w9 = {:.3f}'.\

#       format(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9))

# print('二乗誤差 = {:.3f}'.format(squared_error))
#----------ホールドアウト法の実装(テストデータはランダム選択)----------

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) 
# ----------学習用データを使って線形回帰モデルを学習----------

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train, y_train)



#----------学習用データに対する予測を実行----------

y_pred_train = regr.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
#----------Ridgeを実施してMAEを計測----------

ridge =  Ridge(alpha=0.9).fit(X_train, y_train)

#----------学習用データに対する予測を実行----------

y_pred_train = ridge.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
#----------Lassoを実施してMAEを計測----------

lasso =  Lasso(alpha=0.01).fit(X_train, y_train)

#----------学習用データに対する予測を実行----------

y_pred_train = lasso.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
#----------ElasticNetを実施してMAEを計測----------

elasticnet = ElasticNet(alpha=0.01).fit(X_train, y_train)

#----------学習用データに対する予測を実行----------

y_pred_train = elasticnet.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
# def plot_approximation(est, ax, label=None):

#     """Plot the approximation of ``est`` on axis ``ax``. """

#     ax.plot(X_train, y_train, color='green')

#     ax.scatter(X, y, s=10)

#     ax.plot(X_train, est.predict(X_train[:, np.newaxis]), color='red', label=label)

#     ax.set_ylim((-2, 2))

#     ax.set_xlim((0, 1))

#     ax.set_ylabel('y')

#     ax.set_xlabel('x')

#     ax.legend(loc='upper right')  #, fontsize='small')



# def plot_coefficients(est, ax, label=None, yscale='log'):

#     coef = est.steps[-1][1].coef_.ravel()

#     if yscale == 'log':

#         ax.semilogy(np.abs(coef), marker='o', label=label)

#         ax.set_ylim((1e-1, 1e8))

#     else:

#         ax.plot(np.abs(coef), marker='o', label=label)

#     ax.set_ylabel('abs(coefficient)')

#     ax.set_xlabel('coefficients')

#     ax.set_xlim((0, 9))

    



# fig, ax_rows = plt.subplots(4, 2, figsize=(8, 10))

# degrees = [0, 1, 3, 9]#degreeの値を4つ指定する

# for ax_row, degree in zip(ax_rows, degrees):

#     ax_left, ax_right = ax_row

#     est = make_pipeline(PolynomialFeatures(degree), LinearRegression())

#     est.fit(X, y)

#     plot_approximation(est, ax_left, label='degree=%d' % degree)

#     plot_coefficients(est, ax_right,yscale=None)

    

# plt.tight_layout()
#----------テストデータに対する予測を実行----------

y_pred_test = regr.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
#----------テストデータに対する予測を実行(ridge)----------

y_pred_test = ridge.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
#----------テストデータに対する予測を実行(lasso)----------

y_pred_test = lasso.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
#----------テストデータに対する予測を実行(elasticnet)----------

y_pred_test = elasticnet.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )