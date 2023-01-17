%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/measurements.csv")
data.head()
data.info()
data["temp_inside"].value_counts()
print(data["temp_inside"].isnull().sum())
print(data["refill liters"].isnull().sum())
# "temp_inside"の前処理
temp_temp_inside = data["temp_inside"].dropna()
temp_temp_inside = temp_temp_inside.apply(lambda x: x.replace(",", "."))
temp_temp_inside = temp_temp_inside.astype(np.float)
temp_temp_inside_avg = temp_temp_inside.mean()
data["temp_inside"] = temp_temp_inside
data["temp_inside"] = data["temp_inside"].fillna(temp_temp_inside_avg)
# "refill_liters"の前処理
temp_refill_liters = data["refill liters"].dropna()
temp_refill_liters = temp_refill_liters.apply(lambda x: x.replace(",", "."))
temp_refill_liters = temp_refill_liters.astype(np.float)
temp_refill_liters_avg = temp_refill_liters.mean()
data["refill liters"] = temp_refill_liters
data["refill liters"] = data["refill liters"].fillna(temp_refill_liters_avg)
# "distance","consume"の前処理
data["distance"] = data["distance"].apply(lambda x: x.replace(",", "."))
data["distance"] = data["distance"].astype(np.float)

data["consume"] = data["consume"].apply(lambda x: x.replace(",", "."))
data["consume"] = data["consume"].astype(np.float)
def jisaku1(x):
    if x == "SP98":
        return 1
    else:
        return 0
data["SP98"] = data["gas_type"].apply(lambda x: jisaku1(x))
print(data.info())
data.head()
data.describe()
pd.plotting.scatter_matrix(data, figsize=(10,10))
plt.show()
sns.heatmap(data.corr())
plt.show()
data.corr()
# 係数を求める
y = data["consume"].values
X = data[['speed', 'temp_outside', 'rain']].values
regr = LR(fit_intercept=True)
regr.fit(X, y)

# 重みを取り出す
w0 = regr.intercept_
w1 = regr.coef_[0]
w2 = regr.coef_[1]
w3 = regr.coef_[2]
x1 = data['speed'].values
x2 = data['temp_outside'].values
x3 = data['rain'].values

# 重みと二乗誤差の確認
y_est = w0 + w1 * x1 + w2 * x2 + w3 * x3
squared_error = 0.5 * np.sum((y - y_est) ** 2)
root_squared_error = np.sqrt(squared_error)
absoluted_error = 0.5*np.sum(abs(y - y_est)) 

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}'.format(w0, w1, w2, w3))
print('二乗誤差(MSE) = {:.3f}'.format(squared_error))
print('平方根二乗誤差(RMSE) = {:.3f}'.format(root_squared_error))
print('絶対値誤差(MAE) = {:.3f}'.format(absoluted_error))
