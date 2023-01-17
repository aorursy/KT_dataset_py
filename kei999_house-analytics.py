import pandas as pd
df = pd.read_csv("../input/kc_house_data.csv")
df.head()

import matplotlib.pyplot as plt

y_var = "price"
X_var = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront"]
df_tmp = df[[y_var]+X_var]
pd.plotting.scatter_matrix(df_tmp,alpha=0.3,s=10, figsize=(10,10))#散布図の作成
plt.show()#グラフをここで描画させるための行

y_var = "price"
X_var = ["view","condition","grade","sqft_above","sqft_basement","yr_built"]
df_tmp = df[[y_var]+X_var]
pd.plotting.scatter_matrix(df_tmp,alpha=0.3,s=10, figsize=(10,10))
plt.show()

y_var = "price"
X_var = ["yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15"]
df_tmp = df[[y_var]+X_var]
pd.plotting.scatter_matrix(df_tmp,alpha=0.3,s=10, figsize=(10,10))
plt.show()

df.describe ()

from sklearn.model_selection import train_test_split

# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(df[["sqft_living"]], df["price"], test_size=0.2, random_state=1234)

from sklearn import linear_model 
# 学習
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )


