import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型

from sklearn.model_selection import train_test_split # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库

import math #数学库
data = pd.read_csv('../input/kc_house_data.csv')

data
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors', 'grade']]

# X = data[['sqft_living', 'grade']]

Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)

print(xtrain.shape)

print(ytrain.shape)

print(xtest.shape)

print(ytest.shape)
xtrain = np.asmatrix(xtrain)

xtest = np.asmatrix(xtest)

ytrain = np.ravel(ytrain)

ytest = np.ravel(ytest)
plt.scatter(X['floors'], Y, s = 1)

plt.show()
plt.scatter(X['bathrooms'], Y, s = 1)

plt.show()
plt.scatter(X['bedrooms'], Y, s = 1)

plt.show()
plt.scatter(X['grade'], Y, s = 1)

plt.show()
plt.scatter(X['sqft_living'], Y, s = 1)

plt.show()
X['sqft_living'].hist(bins=25)

plt.show()
model = LinearRegression()

model.fit(xtrain, ytrain)

print(model.coef_.shape)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价

demo = np.array([3,2,2500,2,3])

model.predict([demo])
pred_y = model.predict(xtrain)

print(pred_y.shape, ytrain.shape)

np.sum((ytrain - pred_y) **2)/len(ytrain)

metrics.mean_squared_error(ytrain, pred_y)
np.sum(abs(ytrain - pred_y)/ytrain)/len(ytrain)
predtest = model.predict(xtest)

((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)