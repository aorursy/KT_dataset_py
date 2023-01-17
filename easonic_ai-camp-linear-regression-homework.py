import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型

from sklearn.model_selection import train_test_split # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库

import math #数学库

%matplotlib inline
data = pd.read_csv("../input/kc_house_data.csv")

data.head(7)
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors']]

y = data['price']

X.head(7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
xtrain = np.asmatrix(X_train)

xtest = np.asmatrix(X_test)

ytrain = np.ravel(y_train)

ytest = np.ravel(y_test)
idx = 1



plt.figure(figsize=(16, 8))

for feature, vals in X.iteritems():

#     print('feature:', feature)

#     print('val:\n', vals)

    plt.subplot(2, 2, idx)

    plt.scatter(X[feature], y)

    plt.title(feature)

    idx += 1
idx = 1



plt.figure(figsize=(16, 8))

for feature, vals in X.iteritems():

#     print('feature:', feature)

#     print('val:\n', vals)

    plt.subplot(2, 2, idx)

    X[feature].hist()

    plt.title(feature)

    idx += 1
model = LinearRegression()

model.fit(X_train, y_train)

print('coef:{0}, intercept:{1}'.format(model.coef_,model.intercept_))
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价

model.predict([[3,2,2500,2]])
pred = model.predict(X_train)

((pred-y_train)*(pred-y_train)).sum() / len(y_train)
(abs(pred-y_train)/y_train).sum() / len(y_train)
predtest = model.predict(X_test)

((predtest-y_test)*(predtest-y_test)).sum() / len(y_test)
(abs(predtest-y_test)/y_test).sum() / len(y_test)