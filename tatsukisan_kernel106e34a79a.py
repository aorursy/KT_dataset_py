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

display(df_car.head())
#----------散布図行列を確認----------

# pd.plotting.scatter_matrix(df_car, figsize=(18,18))

# plt.show()
#----------相関係数をヒートマップにして可視化----------

# fig, ax = plt.subplots(figsize=(16, 8 )) 

# sns.heatmap(df_car.corr(), annot=True)

# plt.show()
X = df_car.drop(columns='consume').values

y = df_car['consume'].values.reshape(-1,1) 
#----------ホールドアウト法の実装(テストデータはランダム選択)----------

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234) 
#==========学習データに対するMAEを計算（訓練誤差の評価）==========

regr = LinearRegression()

regr.fit(X_train, y_train)



y_pred_train = regr.predict(X_train)



mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
# ==========テストデータに対するMAEを計算（汎化誤差の評価）==========

y_pred_test = regr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
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
#==========正規化の実行（学習用データ）==========

mms = MinMaxScaler()

X_train_mms = mms.fit_transform(X_train)
#==========正規化実行済みデータのヒストグラム（学習用データ）==========

fig = plt.figure(figsize=(16, 4 ))

ax0 = fig.add_subplot(1, 4, 1)

ax0.hist(X_train_mms[:, 0])

ax0.set_title('distance')



ax1 = fig.add_subplot(1, 4, 2)

ax1.hist(X_train_mms[:, 1])

ax1.set_title('speed')



ax2 = fig.add_subplot(1, 4, 3)

ax2.hist(X_train_mms[:, 2])

ax2.set_title('temp_inside')



ax3 = fig.add_subplot(1, 4, 4)

ax3.hist(X_train_mms[:, 3])

ax3.set_title('temp_outside')
#========正規化したデータで線形回帰を実行========

regr_mms = LinearRegression()

regr_mms.fit(X_train_mms, y_train)



y_pred_train = regr_mms.predict(X_train_mms)



mae = mean_absolute_error(y_train, y_pred_train)

print("MAE = %s"%round(mae,3) )
# ==========テストデータに対するMAEを計算（汎化誤差の評価）==========

mms = MinMaxScaler()

X_test_mms = mms.fit_transform(X_test)

y_pred_test = regr_mms.predict(X_test_mms)

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
#========標準化の実行（学習用データ）========

stdsc = StandardScaler()

X_train_stand = stdsc.fit_transform(X_train)
#==========標準化実行済みデータのヒストグラム（学習用データ）==========

fig = plt.figure(figsize=(16, 4 ))

ax0 = fig.add_subplot(1, 4, 1)

ax0.hist(X_train_stand[:, 0])

ax0.set_title('distance')



ax1 = fig.add_subplot(1, 4, 2)

ax1.hist(X_train_stand[:, 1])

ax1.set_title('speed')



ax2 = fig.add_subplot(1, 4, 3)

ax2.hist(X_train_stand[:, 2])

ax2.set_title('temp_inside')



ax3 = fig.add_subplot(1, 4, 4)

ax3.hist(X_train_stand[:, 3])

ax3.set_title('temp_outside')
#========標準化したデータで線形回帰を実行========

regr_stand = LinearRegression()

regr_stand.fit(X_train_stand, y_train)



y_pred_stand_train = regr_stand.predict(X_train_stand)



mae = mean_absolute_error(y_train, y_pred_stand_train)

print("MAE = %s"%round(mae,3) )
# ==========テストデータに対するMAEを計算（汎化誤差の評価）==========

stdsc = StandardScaler()

X_test_stand = stdsc.fit_transform(X_test)

y_pred_test = regr_stand.predict(X_test_stand)

mae = mean_absolute_error(y_test, y_pred_test)

print("MAE = %s"%round(mae,3) )
#========Ridgeを実施してMAEを計測========

ridge =  Ridge(alpha=0.9).fit(X_train, y_train)



#----------学習用データに対する予測を実行----------

y_pred_train = ridge.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae_ridge_train = mean_absolute_error(y_train, y_pred_train)

print("MAE_train = %s"%round(mae_ridge_train,3) )
#----------テストデータに対する予測を実行(ridge)----------

y_pred_test = ridge.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae_ridge_test = mean_absolute_error(y_test, y_pred_test)

print("MAE(Ridge_test) = %s"%round(mae_ridge_test,3) )
#----------Lassoを実施してMAEを計測----------

lasso =  Lasso(alpha=0.01).fit(X_train, y_train)

#----------学習用データに対する予測を実行----------

y_pred_train = lasso.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae_lasso_train = mean_absolute_error(y_train, y_pred_train)

print("MAE(Lasso_train) = %s"%round(mae_lasso_train,3) )
#----------テストデータに対する予測を実行(lasso)----------

y_pred_test = lasso.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae_lasso_test = mean_absolute_error(y_test, y_pred_test)

print("MAE(Lasso_test) = %s"%round(mae_lasso_test,3) )
#----------ElasticNetを実施してMAEを計測----------

elasticnet = ElasticNet(alpha=0.01).fit(X_train, y_train)

#----------学習用データに対する予測を実行----------

y_pred_train = elasticnet.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae_ela_train = mean_absolute_error(y_train, y_pred_train)

print("MAE(ElasticNet_train) = %s"%round(mae_ela_train,3) )
#----------テストデータに対する予測を実行(elasticnet)----------

y_pred_test = elasticnet.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae_ela_test = mean_absolute_error(y_test, y_pred_test)

print("MAE(ElasticNet_test) = %s"%round(mae_ela_test,3) )
#========多項式化を実施してMAEを計測========

est = make_pipeline(PolynomialFeatures(2), LinearRegression())

est.fit(X_train, y_train)



#----------学習用データに対する予測を実行----------

y_pred_train = est.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae_pol_train = mean_absolute_error(y_train, y_pred_train)

print("MAE(Polynomial_train) = %s"%round(mae_pol_train,3) )
#----------テストデータに対する予測を実行(est)----------

y_pred_test = est.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae_pol_test = mean_absolute_error(y_test, y_pred_test)

print("MAE(Polynomial_test) = %s"%round(mae_pol_test,3) )
#========多項式化+ElasticNet========

est2 = make_pipeline(PolynomialFeatures(2), ElasticNet(alpha=0.8))

est2.fit(X_train, y_train)



#----------学習用データに対する予測を実行----------

y_pred_train = est2.predict(X_train)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae_pol_ela_train = mean_absolute_error(y_train, y_pred_train)

print("MAE(Polynomial+ElasticNet_train) = %s"%round(mae_pol_ela_train,3) )
#----------テストデータに対する予測を実行(est2)----------

y_pred_test = est2.predict(X_test)



#----------テストデータに対するMAEを計算（汎化誤差の評価）----------

mae_pol_ela_test = mean_absolute_error(y_test, y_pred_test)

print("MAE(Polynomial+ElasticNet_train) = %s"%round(mae_pol_ela_test,3) )
#精度格納用

df_precision = pd.DataFrame(index=['normal_MAE', 'Ridge_MAE','Lasso_MAE','Elasticnet_MAE','R2_pred'], columns=['train', 'test'])

display(df_precision)
from sklearn.feature_selection import RFECV

estimator = LinearRegression(normalize=True)

rfecv = RFECV(estimator, cv=10, scoring='neg_mean_absolute_error')



train_label = df_car["consume"]

train_data = df_car.drop("consume", axis=1)



y = train_label

X = train_data



# fitで特徴選択を実行

rfecv.fit(X, y)
# ========特徴のランキングを表示（1が最も重要な特徴）========

print('Feature ranking: \n{}'.format(rfecv.ranking_))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
rfecv.support_
remove_idx = ~rfecv.support_

remove_idx
remove_feature =train_data.columns[remove_idx]

remove_feature
selected_train_data = train_data.drop(remove_feature, axis=1)

selected_train_data.head()
regr_stand2 = LinearRegression(normalize=True)

regr_stand2.fit(selected_train_data, train_label)



y_pred_train = regr_stand2.predict(selected_train_data)



mae = mean_absolute_error(train_label, y_pred_train)

print("MAE = %s"%round(mae,3) )
#========ステップワイズ＋多項式化+ElasticNet========

est3 = make_pipeline(PolynomialFeatures(2), ElasticNet(alpha=0.8))

est3.fit(selected_train_data, train_label)



#----------学習用データに対する予測を実行----------

y_pred_train2 = est3.predict(selected_train_data)



#----------学習データに対するMAEを計算（訓練誤差の評価）----------

mae_pol_ela_train2 = mean_absolute_error(train_label, y_pred_train2)

print("MAE(Polynomial+ElasticNet_train+STEP wise) = %s"%round(mae_pol_ela_train2,3) )