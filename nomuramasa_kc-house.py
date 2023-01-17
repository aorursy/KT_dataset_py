import pandas as pd
df = pd.read_csv('../input/kc_house_data.csv')
df.head()
# df_large = df[df['sqft_living'] > 10000]  # 家の平方フィート（面積？）が広い
# df_large
import matplotlib.pyplot as plt

#今回予測したい「値段」
y_var = 'price' 

#関係ありそうな
x_var = ['sqft_living', 'sqft_basement', 'yr_built', 'bedrooms', 'bathrooms', 'sqft_living15', 'grade'] 

df_scatter = df[[y_var] + x_var] #スキャッター

pd.plotting.scatter_matrix(df_scatter, alpha=0.8, s=10, figsize=(10,10))
# df["yr_built"] = 1 / df["yr_built"] #傾きが逆に相関する場合

plt.show()
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 説明変数（Numpyの配列） 関係ありそうな値？
X = df[['sqft_living']].values

# 目的変数（Numpyの配列）
Y = df['price'].values

lr.fit(X, Y)            
print('coefficient = ', lr.coef_[0]) # 説明変数の係数を出力
print('intercept = ', lr.intercept_) # 切片を出力
plt.plot(df['sqft_living'],df['sqft_living'] * lr.coef_[0] + lr.intercept_,"r-")
plt.scatter(df['sqft_living'],df['price'])
from sklearn.model_selection import train_test_split

x = df[['sqft_living']].values
y = df['price'].values


# 利用できるデータのうち、学習用を8割、テスト用を2割にする
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

print("len(sqft_living)=",len(x))
print("len(price)=",len(y))
print("")
print("len(x_train)=",len(x_train))
print("len(y_train)=",len(y_train))
print("")
print("len(x_test)=",len(x_test))
print("len(y_test)=",len(y_test))
print("")
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 学習用データで線形モデルに学習させる。
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(x_train, y_train)

# 予測データを使用してMSE（平均二乗誤差）を求める。
y_pred = regr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
# MSE=14654074192.893
# RMSE=121054.014
# MAE=94043.912