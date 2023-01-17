# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#ライブラリのインポート

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数

#データセットのインポート

car_fuel = pd.read_csv("../input/measurements.csv")
car_fuel.head()
car_fuel.info()
for column_name in (['distance','consume','temp_inside']):    

    car_fuel_new = car_fuel.loc[:,column_name]

    car_fuel_new = car_fuel_new.str.replace(",",".").astype(float)

    car_fuel.loc[:,column_name] = car_fuel_new

car_fuel.head()
car_fuel.info()
# 散布図行列を書いてみる

pd.plotting.scatter_matrix(car_fuel, figsize=(10,10))

plt.show()

#目的変数consume に有効そうな変数はdistance と spped?
# temp_inside 欠損値をtemp_insideの平均値で置き換える

car_fuel.loc[(car_fuel.temp_inside.isnull()),"temp_inside"] = car_fuel.loc[:,"temp_inside"].mean()

car_fuel.info()
# gas_type を数字に置き換える

car_fuel.loc[car_fuel["gas_type"]=="E10","gas_type"]=0

car_fuel.loc[car_fuel["gas_type"]=="SP98","gas_type"]=1

car_fuel.head()
# 不要なデータを削除する

# specials データ量不足＋扱いにくい

# refill liters データ量の不足

# refill gas データ量の不足＋gas_typeとの重複

drop_columns = ["specials","refill liters","refill gas"]

car_fuel = car_fuel.drop(drop_columns, axis = 1)

car_fuel.head()
# 相関係数を確認

car_fuel.corr()
# 係数を求める

y = car_fuel["consume"].values

X = car_fuel[['distance', 'speed', 'temp_inside', 'temp_outside', 'gas_type', 'AC', 'rain', 'sun']].values

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

w8 = regr.coef_[7]
x1 = car_fuel['distance'].values

x2 = car_fuel['speed'].values

x3 = car_fuel['temp_inside'].values

x4 = car_fuel['temp_outside'].values

x5 = car_fuel['gas_type'].values

x6 = car_fuel['AC'].values

x7 = car_fuel['rain'].values

x8 = car_fuel['sun'].values



# 重みと二乗誤差の確認

y_est = w0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6

squared_error = 0.5 * np.sum((y - y_est) ** 2)

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

print('二乗誤差 = {:.3f}'.format(squared_error))
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