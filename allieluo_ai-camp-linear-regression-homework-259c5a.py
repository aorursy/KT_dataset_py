import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型
from sklearn.model_selection import train_test_split # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import math #数学库
from scipy import stats
data = pd.read_csv('../input/kc_house_data.csv')
data.shape
data.head()
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view']]
Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
xtrain[0:2]
ytrain[0:1]
plt.scatter(X['sqft_living'], Y)
plt.show()
X['sqft_living'].hist()
plt.show()
model = LinearRegression()
model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
stats.describe(xtrain)
#   bedrooms	-38694.943285
#1	bathrooms	3778.913999
#2	sqft_living	269.320260
#3	sqft_lot	-0.305805
#4	floors	14787.378052
#5	waterfront	582134.349947
#6	view, 74319.720507
model.predict([[3.000e+00, 1.500e+00, 1.260e+03, 1.035e+04, 1.000e+00,
         0.000e+00, 0.000e+00]])
# view +1
model.predict([[3.000e+00, 1.500e+00, 1.260e+03, 1.035e+04, 1.000e+00,
         0.000e+00, 1.000e+00]])
#bedrooms +1
model.predict([[4.000e+00, 1.500e+00, 1.260e+03, 1.035e+04, 1.000e+00,
         0.000e+00, 0.000e+00]])
#calculate MSE of training data
pred = model.predict(xtrain)
((pred-ytrain)*(pred-ytrain)).sum() / len(ytrain)
#calcualte average relative error of training data
(abs(pred-ytrain)/ytrain).sum() / len(ytrain)
#calculate MSE of test data
pred1 = model.predict(xtest)
((pred1-ytest)*(pred1-ytest)).sum() / len(ytest)
#calcualte average relative error of test data
(abs(pred1-ytest)/ytest).sum() / len(ytest)
