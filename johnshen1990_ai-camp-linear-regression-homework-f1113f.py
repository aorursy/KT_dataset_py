import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
data = pd.read_csv("../input/kc_house_data.csv")
data.dtypes
origin_X_train = data[['bedrooms','bathrooms','sqft_living','floors']]
origin_y_train = data['price']
X_train, X_vali, y_train, y_vali = train_test_split(origin_X_train, origin_y_train, test_size = 0.2, random_state = 0)
X_train = np.asmatrix(X_train)
X_vali = np.asmatrix(X_vali)
y_train = np.ravel(y_train)
y_vali = np.ravel(y_vali)
plt.scatter(origin_X_train["sqft_living"], origin_y_train)
plt.show()
origin_X_train["sqft_living"].hist()
plt.show()
# 建立线性回归模型
model = LinearRegression()
# 训练模型
model.fit(X_train, y_train)
pd.DataFrame(list(zip(origin_X_train.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价
model.predict([[3, 2, 2500, 2]])

predictions_train = model.predict(X_train)
mse_train = ((predictions_train - y_train) ** 2).sum() / len(X_train)
print("mse_train=", mse_train)
md_train = (abs(predictions_train - y_train) / y_train).sum() / len(X_train)
print("md_train=", md_train)
predictions_vali = model.predict(X_vali)
mse_vali = ((predictions_vali - y_vali) ** 2).sum() / len(X_vali)
print("mse_vali=", mse_vali)
md_vali = (abs(predictions_vali - y_vali) / y_vali).sum() / len(X_vali)
print("md_vali=", md_vali)
