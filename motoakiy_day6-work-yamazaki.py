%matplotlib inline 
#グラフをnotebook内に描画させるための設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
df_data = pd.read_csv("../input/kc_house_data.csv")
df_data["price"] = df_data["price"] / 10**6 #単位を100万ドルにしておく
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
df_data.corr()
y_var = "price"
X_var = ["sqft_living","grade","sqft_above","sqft_living15","bathrooms","view"]
df = df_data[[y_var]+ X_var]
display(df.head())

pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(7,7))#散布図の作成
plt.show()#グラフをここで描画させるための行
y_var = "price"
X_var = ["sqft_living","grade","sqft_above","sqft_living15","bathrooms"]
df = df_data[[y_var]+ X_var]
display(df.head())

pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(7,7))#散布図の作成
plt.show()#グラフをここで描画させるための行
# 欠損値のデータが含まれているかどうか確認する
pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])
# Nanの除去
df = df_data.dropna()

# scikit learnの入力形式に変換する
X = df[X_var].as_matrix()
y = df[y_var].values

# 学習
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X, y)

print("決定係数=",regr.score(X,y))
regr.intercept_, regr.coef_
from sklearn.model_selection import train_test_split

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print("X_train")
print(X_train)
print("")
print("X_test")
print(X_test)
print("")
print("y_train")
print(y_train)
print("")
print("y_test")
print(y_test)
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )

y = df_data["price"].values
X = df_data[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']].values

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

kf = KFold(n_splits=5, random_state=1234, shuffle=True)
kf.get_n_splits(X, y)

df_result = pd.DataFrame()

for train_index, test_index in kf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    regr = LinearRegression(fit_intercept=True)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    df = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
    df_result = pd.concat([df_result, df], axis=0)

# 評価指標の算出
y_test = df_result["y_test"]
y_pred = df_result["y_pred"]
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )

