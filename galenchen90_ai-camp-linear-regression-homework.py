import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
data = pd.read_csv("../input/kc_house_data.csv")
data.head(10)
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors','view','yr_renovated']]
Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
print(type(xtrain))
xtrain.head()

print(type(ytrain))
ytrain.head()
xtrain = np.asmatrix(xtrain)   #只需要他们的数字
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
xtrain  
ytrain #所有的房价
plt.scatter(X['sqft_living'], Y)
plt.show()
X['sqft_living'].hist()
plt.show()

plt.scatter(X['bedrooms'], Y)
plt.show()
model = LinearRegression()
model.fit(xtrain, ytrain)
model
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.predict([[3,2,2500,2,1,2000]])
model.intercept_
model.coef_
pd.DataFrame(list(zip(X.columns,np.transpose(model.coef_))))
pred = model.predict(xtrain)
((pred-ytrain)*(pred-ytrain)).sum() / len(ytrain)
(abs(pred-ytrain)/ytrain).sum() / len(ytrain)

predtest = model.predict(xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)




