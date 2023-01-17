import numpy as np

import pandas as pd #读CSV

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import math
data = pd.read_csv('../input/kc_house_data.csv')

data[:3]
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors']]

Y = data['price']
#split train 

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 1/3, random_state=0)
plt.scatter(X['sqft_living'], Y)#面积是决定房价的因素

plt.show()
X['sqft_living'].hist()

plt.show()
model = LinearRegression()

model.fit(xtrain, ytrain)#fit 完成，就train好了
print(model.coef_)#Estimated coefficients for the linear regression problem

print(model.intercept_)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))#transpose 转置

#不要根据数字本身解读哪个重要
#一个房子， 3个卧室， 2个卫生间， 2500sqft， 2层楼

model.predict([[3, 2, 2500, 3]])
pred = model.predict(xtrain)

((pred - ytrain) * (pred - ytrain)).sum()/len(ytrain)
(abs(pred - ytrain) / ytrain).sum() / len(ytrain)
predtest = model.predict(xtest)

((predtest - ytest) * (predtest - ytest).sum() /len(ytest))
(abs(predtest - ytest) / ytest).sum() / len(ytest)