import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
data = pd.read_csv('../input/kc_house_data.csv')

data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors']]
Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, test_size = 1/3, random_state = 0
    )
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
plt.scatter(data['sqft_living'], data['price'])

data['sqft_living'].hist()

model = LinearRegression()
model.fit(xtrain, ytrain)
model.coef_
np.transpose(model.coef_)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价
model.predict([[3, 2, 2500, 2]])

Y_pred = model.predict(xtrain)
Y_test = ytrain
mse = sum((Y_pred - Y_test) ** 2) / len(Y_pred)
mse
avg_mse = sum( abs(Y_pred - Y_test)/ Y_test ) / len(Y_pred)
avg_mse
predtest = model.predict(xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)
