import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
data = pd.read_csv('../input/kc_house_data.csv')
print(data.head())
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors']]
print (X.head())
Y = data['price']
print (Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
X_train = np.asmatrix(X_train)
X_test = np.asmatrix(X_test)
Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)
plt.scatter(X['sqft_living'], Y)
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()
X['sqft_living'].hist(bins = 20)
model = LinearRegression()
model.fit(X_train, Y_train)
print(model.coef_)
print(X.columns)
print(list(zip(X.columns, model.coef_)))
pd.DataFrame(list(zip(X.columns, model.coef_)))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价
model.predict([[3, 2, 2500, 2]])
X_train_predict = model.predict(X_train)
((X_train_predict - Y_train) ** 2.).sum() / len(Y_train)
(abs(X_train_predict - Y_train) / Y_train).sum() / len(Y_train)
X_test_predict = model.predict(X_test)
((X_test_predict-Y_test) ** 2.).sum() / len(Y_test)
(abs(X_test_predict-Y_test) / Y_test).sum() / len(Y_test)
