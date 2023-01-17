import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 各项指标
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
data = pd.read_csv('../input/kc_house_data.csv')
data
data.dtypes
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors']]
Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 1/3, random_state = 0) 
xtrain
xtrain = np.asmatrix(xtrain)#改为纯数字矩阵(原来的表头和行号都没了)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain) #改为array(一列)(原来的表头和行号都没了)
ytest = np.ravel(ytest)
xtrain
ytrain
plt.scatter(X['sqft_living'], Y) 
plt.show()
X['sqft_living'].hist()
plt.show()
plt.scatter(X['bedrooms'], Y)
plt.show()
model = LinearRegression()
model.fit(xtrain, ytrain) #模型已经训练好了
#一个房子，3个卧室，2个卫生间，2500sqft, 2层楼
model.predict([[3, 2, 2500, 2]])
model.coef_ #模型参数
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))) 
model.intercept_  #截矩
model.predict([[3, 2, 2000, 3]]) - model.predict([[3, 2, 2000, 2]]) #验证模型系数
pred = model.predict(xtrain)
((pred - ytrain) ** 2).sum() / len(ytrain) 
(abs(pred - ytrain) / ytrain).sum() / len(ytrain)
predtest = model.predict(xtest)
((predtest - ytest) ** 2).sum() / len(ytest) 
(abs(predtest - ytest) / ytest).sum() / len(ytest)
