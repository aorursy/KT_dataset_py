import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math
data = pd.read_csv('../input/kc_house_data.csv')
data
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors']]
Y = data['price']

print(type(X))
print(type(Y))
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
print(xtrain.head())
print(ytrain.head())
print(xtest.head())
print(ytest.head())
xtrain = np.asmatrix(xtrain)
print(xtrain)
xtest = np.asmatrix(xtest)
print(xtest)
ytrain = np.ravel(ytrain)
print(ytrain)
ytest = np.ravel(ytest)
print(ytest)


plt.scatter(X['sqft_living'], Y)
plt.show()
X['sqft_living'].hist()
plt.show()
model = LinearRegression()
model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价
model.predict([[3, 2, 2500, 2]])
pred = model.predict(xtrain)
((pred - ytrain) * (pred - ytrain)).sum() / len(ytrain)
(abs(pred - ytrain) / ytrain).sum() / len(ytrain)
predtest = model.predict(xtest)
((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
(abs(predtest-ytest)/ytest).sum() / len(ytest)
