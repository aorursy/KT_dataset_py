%matplotlib inline 
#グラフをnotebook内に描画させるための設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
df_data = pd.read_csv("../1_data/kc_house_data.csv")
df_data["price"] = df_data["price"] / 10**6 #単位を100万ドルにしておく
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
y_var = "price"
X_var = ["sqft_living","sqft_lot","sqft_living15","sqft_lot15","bedrooms","bathrooms"]
df = df_data[[y_var]+ X_var]
display(df.head())

pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(7,7))#散布図の作成
plt.show()#グラフをここで描画させるための行
df_data.drop(["id","date"],inplace=True, axis=1)
import statsmodels.api as sm #線形回帰分析と同時にAICを計算してくれる
import itertools

count = 1
for i in range(21):
    combi = itertools.combinations(df_data.drop("price",axis=1).columns, i+1) #組み合わせを求める
    for v in combi:
        y = df_data["price"]
        X = sm.add_constant(df_data[list(v)])
        model = sm.OLS(y, X).fit()
        if count == 1:
            min_aic = model.aic
            min_var = list(v)
        if min_aic > model.aic:
            min_aic = model.aic
            min_var = list(v)
        count += 1
        print("AIC:",round(model.aic), "変数:",list(v))
print("====minimam AIC====")
print(min_var,min_aic)

y_var = "price"
X_var = ["sqft_living","grade","yr_built","lat"]
df = df_data[[y_var]+ X_var]
display(df.head())

pd.plotting.scatter_matrix(df,alpha=0.3,s=10, figsize=(7,7))#散布図の作成
plt.show()#グラフをここで描画させるための行

df_data[df_data["sqft_living"].isnull()]
df_data[df_data["grade"].isnull()]
df_data[df_data["yr_built"].isnull()]
df_data[df_data["lat"].isnull()]




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
np.mean(np.square(y_test - y_pred))
np.mean(np.absolute(y_test - y_pred))