import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型

from sklearn.model_selection import train_test_split # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库

import math #数学库
data = pd.read_csv('../input/kc_house_data.csv')

print(data.shape)

data.head()
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living', 'floors', 'condition','grade', 'yr_built']]

Y = data['price']



X.head()
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
xtrain = np.asmatrix(xtrain)

xtest = np.asmatrix(xtest)

ytrain = np.ravel(ytrain)

ytest = np.ravel(ytest)
plt.scatter(X['sqft_living'], Y)

plt.title('price vs sqft_living')

plt.show()



plt.scatter(X['bedrooms'], Y)

plt.title('price vs bedrooms')

plt.show()



plt.scatter(X['bathrooms'], Y)

plt.title('price vs bathrooms')

plt.show()



plt.scatter(X['floors'], Y)

plt.title('price vs floors')

plt.show()

plt.scatter(X['condition'], Y)

plt.title('price vs condition')

plt.show()



plt.scatter(X['grade'], Y)

plt.title('price vs grade')

plt.show()



plt.scatter(X['yr_built'], Y)

plt.title('price vs built')

plt.show()
X['sqft_living'].hist()

plt.show()
model = LinearRegression()

model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价

# model.predict([[3, 2, 2500, 2]])
pred = model.predict(xtrain)

MSE_train = ((pred - ytrain) ** 2).sum() / len(ytrain)

RMSD_train = MSE_train ** 0.5

print(RMSD_train)



ERR_train = (abs(pred - ytrain) / ytrain).sum() / len(ytrain)

print(ERR_train)
pred_test = model.predict(xtest)

MSE_test = ((pred_test - ytest) ** 2).sum() / len(ytest)

RMSD_test = MSE_test ** 0.5

print(RMSD_test)





ERR_test = (abs(pred_test - ytest) / ytest).sum() / len(ytest)

print(ERR_test)