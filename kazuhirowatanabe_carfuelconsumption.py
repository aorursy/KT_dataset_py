import math



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel("../input/car-consume/measurements2.xlsx")

print(df)
df_scatter = df.loc[:,["consume", "distance", "speed", "temp_inside", "temp_outside"]]

pd.plotting.scatter_matrix(df_scatter, diagonal="kde", figsize=(10.0, 10.0))
# gas_type

df[["consume", "gas_type"]].boxplot(by="gas_type")
# AC

df[["consume", "AC"]].boxplot(by="AC")
# rain

df[["consume", "rain"]].boxplot(by="rain")
# sun

df[["consume", "sun"]].boxplot(by="sun")
x = df["distance"].values

y = df["consume"].values



plt.scatter(x, y)

plt.xlabel("distance")

plt.ylabel("consume")

plt.show()
plt.scatter(np.reciprocal(x), y)

plt.xlabel("1/distance")

plt.ylabel("consume")

plt.show()
x = np.reciprocal(x)

regr = LinearRegression()



x_train = x.reshape(-1,1)

regr.fit(x_train, y)
consume_pred = regr.intercept_ + regr.coef_[0] * x



plt.plot(x, y, "o")

plt.xlabel("1/distance")

plt.ylabel("consume")

plt.plot(x, consume_pred)

plt.show()
# 精度評価用の関数

def MAE(pred, reference):

    assert len(pred) == len(reference)

    diff = pred - reference

    diff = np.abs(diff)

    return diff.mean()



def MSE(pred, reference):

    assert len(pred) == len(reference)

    diff = pred - reference

    diff = diff**2

    return diff.mean()



def RMSE(pred, reference):

    return math.sqrt(MSE(pred, reference))
mae = MAE(consume_pred, y)

mse = MSE(consume_pred, y)

rmse = RMSE(consume_pred, y)

print("(consume), (1/distance)")

print("MAE = {0:.3f} (0.01 × L/km)".format(mae))

print("MSE = {0:.3f} (1.0e-4 × L**2/km**2)".format(mse))

print("RMSE = {0:.3f} (0.01 × L/km)".format(rmse))
n_RS = len(df[(df.rain == 1) & (df.sun == 1)])    # 降雨あり かつ 日射あり

n_Rs = len(df[(df.rain == 1) & (df.sun == 0)])    # 降雨あり かつ 日射なし

n_rS = len(df[(df.rain == 0) & (df.sun == 1)])    # 降雨なし かつ 日射あり

n_rs = len(df[(df.rain == 0) & (df.sun == 0)])    # 降雨なし かつ 日射なし



print("        日射あり, 日射なし")

print("降雨あり{0:8d}, {1:8d}".format(n_RS, n_Rs))

print("降雨なし{0:8d}, {1:8d}".format(n_rS, n_rs))
# rain

x = df["rain"].values

y = df["consume"].values



regr = LinearRegression()

x_train = x.reshape(-1,1)

regr.fit(x_train, y)

consume_pred = regr.intercept_ + regr.coef_[0] * x



plt.plot(x, y, "o")

plt.xlabel("rain")

plt.ylabel("consume")

plt.plot(x, consume_pred)

plt.show()



mae = MAE(consume_pred, y)

mse = MSE(consume_pred, y)

rmse = RMSE(consume_pred, y)

print("(consume), (rain)")

print("MAE = {0:.3f} (0.01 × L/km)".format(mae))

print("MSE = {0:.3f} (1.0e-4 × L**2/km**2)".format(mse))

print("RMSE = {0:.3f} (0.01 × L/km)".format(rmse))

# sun

x = df["sun"].values

y = df["consume"].values



regr = LinearRegression()

x_train = x.reshape(-1,1)

regr.fit(x_train, y)

consume_pred = regr.intercept_ + regr.coef_[0] * x



plt.plot(x, y, "o")

plt.xlabel("rain")

plt.ylabel("consume")

plt.plot(x, consume_pred)

plt.show()



mae = MAE(consume_pred, y)

mse = MSE(consume_pred, y)

rmse = RMSE(consume_pred, y)

print("(consume), (sun)")

print("MAE = {0:.3f} (0.01 × L/km)".format(mae))

print("MSE = {0:.3f} (1.0e-4 × L**2/km**2)".format(mse))

print("RMSE = {0:.3f} (0.01 × L/km)".format(rmse))
x = df[["distance", "rain"]].values

y = df["consume"].values

x[:,0] = np.reciprocal(x[:,0])    # distance → 1/distance



x_train = x

regr.fit(x_train, y)



consume_pred = regr.intercept_ + regr.coef_[0] * x[:,0] + regr.coef_[1] * x[:,1]



mae = MAE(consume_pred, y)

mse = MSE(consume_pred, y)

rmse = RMSE(consume_pred, y)

print("(consume), (1/distance, rain)")

print("MAE = {0:.3f} (0.01 × L/km)".format(mae))

print("MSE = {0:.3f} (1.0e-4 × L**2/km**2)".format(mse))

print("RMSE = {0:.3f} (0.01 × L/km)".format(rmse))