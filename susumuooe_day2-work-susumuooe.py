%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数
df_measurements = pd.read_csv("../input/measurements_hosei.csv")
display(df_measurements.head())
df_measurements.info()
df_measurements.describe(include='all')
df_measurements_E10  = df_measurements.query('gas_type == "E10"')
df_measurements_SP98 = df_measurements.query('gas_type == "SP98"')
df_measurements_E10.describe(include='all')
df_measurements_SP98.describe(include='all')
pd.plotting.scatter_matrix(df_measurements_E10, figsize=(15,15))
plt.show()
# 相関係数を確認
df_measurements_E10.corr()
# 相関係数をヒートマップにして可視化
sns.heatmap(df_measurements_E10.corr())
plt.show()
pd.plotting.scatter_matrix(df_measurements_SP98, figsize=(15,15))
plt.show()
# 相関係数を確認
df_measurements_SP98.corr()
# 相関係数をヒートマップにして可視化
sns.heatmap(df_measurements_SP98.corr())
plt.show()
# 係数を求める
y = df_measurements_E10["consume"].values
X = df_measurements_E10[["distance", "speed", "temp_inside", "temp_outside", "AC" ,"rain" ,"sun"]].values
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

# 変数の設定
x1 = df_measurements_E10["distance"]
x2 = df_measurements_E10["speed"]
x3 = df_measurements_E10["temp_inside"]
x4 = df_measurements_E10["temp_outside"]
x5 = df_measurements_E10["AC"]
x6 = df_measurements_E10["rain"]
x7 = df_measurements_E10["sun"]
y = df_measurements_E10["consume"]

# 重みと二乗誤差の確認
y_est = w0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7
squared_error = 0.5 * np.sum((y - y_est) ** 2)
print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}, w7 = {:.3f}, 二乗誤差 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6, w7, squared_error))

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
# 係数を求める
y = df_measurements_SP98["consume"].values
X = df_measurements_SP98[["distance", "speed", "temp_inside", "temp_outside", "AC" ,"rain" ,"sun"]].values
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

# 変数の設定
x1 = df_measurements_SP98["distance"]
x2 = df_measurements_SP98["speed"]
x3 = df_measurements_SP98["temp_inside"]
x4 = df_measurements_SP98["temp_outside"]
x5 = df_measurements_SP98["AC"]
x6 = df_measurements_SP98["rain"]
x7 = df_measurements_SP98["sun"]
y = df_measurements_SP98["consume"]

# 重みと二乗誤差の確認
y_est = w0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7
squared_error = 0.5 * np.sum((y - y_est) ** 2)
print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}, w7 = {:.3f}, 二乗誤差 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6, w7, squared_error))

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
