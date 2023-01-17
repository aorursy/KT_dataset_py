import numpy as np 
import pandas as pd   # csv常用库
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import matplotlib.pyplot as plt
import math
# 从../input/kc_house_data.csv文件中读入数据
data = pd.read_csv("../input/kc_house_data.csv")
data
data.dtypes
# 获得自变量X和因变量Y
X = data[['bedrooms','bathrooms','sqft_living','floors', 'yr_built', 'sqft_lot', 'view']]
Y = data['price']
# trainset : testset = 2:1
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
# 观察房价和生活面积的关系
plt.scatter(X['sqft_living'], Y)
plt.show()
# 观察生活面积分布
X['sqft_living'].hist()
plt.show()
X['bedrooms'].hist()
plt.show()
# To train the model, using xtrain, ytrain
model = LinearRegression()
model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
model.coef_
# 3 bedroom，2 bathroom，2500sqft，2 fl, 1985, 放入bias 1
model.predict([[3,2,2500,2, 1985, 2140, 1]])
# 均方差：MSE
pred = model.predict(xtrain)
((pred-ytrain) ** 2).sum() / len(ytrain)
# 平均相对误差
(abs(pred-ytrain)/ytrain).sum() / len(ytrain)
'''
MSE on Test dataset
'''
predtest = model.predict(xtest)
((predtest-ytest) ** 2).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)

