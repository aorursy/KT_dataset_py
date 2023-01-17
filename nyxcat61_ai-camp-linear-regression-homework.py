import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型

from sklearn.model_selection import train_test_split # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库

import math #数学库
data = pd.read_csv('../input/kc_house_data.csv')

data.head(5)
data.dtypes
data.isnull().values.any()
X = data[['bedrooms','bathrooms','sqft_living','floors']]

Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
xtrain = np.asmatrix(xtrain)

xtest = np.asmatrix(xtest)

ytrain = np.ravel(ytrain)

ytest = np.ravel(ytest)
plt.hist(Y/1000, 50)

plt.xlabel('House price (k)')

plt.ylabel('Count')

plt.title('Distribution of house price')

plt.show()

# 对label取对数，使得房价呈正态分布

plt.hist(np.log(Y/1000), 50)

plt.xlabel('Log[House price (k)]')

plt.ylabel('Count')

plt.title('Distribution of log(house price)')

plt.show()
plt.scatter(data['sqft_living'], Y)

plt.show()
plt.hist(data['sqft_living'], 50)

plt.show()
model = LinearRegression()

model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价

model.predict([[3, 2, 2500, 2]])
pred_train = model.predict(xtrain)

np.sum((pred_train - ytrain) ** 2) / len(ytrain)
np.sum(abs(pred_train - ytrain) / ytrain) / len(ytrain)
predtest = model.predict(xtest)

((predtest-ytest)*(predtest-ytest)).sum() / len(ytest)
data['log_sqft_living'] = np.log(data['sqft_living'])

data['log_price'] = np.log(data['price'])



plt.scatter(data['log_sqft_living'], data['log_price'])

plt.show()
X = data[['bedrooms','bathrooms','log_sqft_living','floors']]

Y = data['log_price']



xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)



xtrain = np.asmatrix(xtrain)

xtest = np.asmatrix(xtest)

ytrain = np.ravel(ytrain)

ytest = np.ravel(ytest)



model = LinearRegression()

model.fit(xtrain, ytrain)



# log房价还原为房价，计算mse

log_pred_test = model.predict(xtest)

pred_test = np.exp(log_pred_test)

ytest = np.exp(ytest)

mse_test = np.sum((pred_test - ytest) ** 2) / len(ytest)

print(mse_test)
import copy



features = ['bedrooms', 'bathrooms', 'log_sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement',]



X = data[features]

Y = data['log_price']



# split train and valid

xtrain, xvalid, ytrian, yvalid = train_test_split(X, Y, test_size=1/3, random_state=0)



mses = []

best_feature_sets = []



best_features = []

for i in range(len(features)):

    best_mse = np.inf

    f = None

    for f in features:

        current_features = best_features + [f]

        xtrain_subset = xtrain[current_features]

        model = LinearRegression()

        model.fit(xtrain_subset, ytrain)

        log_pred_valid = model.predict(xvalid[current_features])

        pred_valid = np.exp(log_pred_valid)

        mse = metrics.mean_squared_error(np.exp(yvalid), pred_valid)

#         print('current feature ', current_features)

#         print('current mse: ', mse)

        if mse < best_mse:

            best_mse = mse

            best_f = f

    features.remove(best_f)

    best_features.append(best_f)

    

    best_features_copy = best_features.copy()

    best_feature_sets.append(best_features_copy)

    mses.append(best_mse)

    print('best mse: ', best_mse, 'best features: ', best_features)        
plt.plot(mses)

plt.show()

min_idx = np.argmin(np.array(mses))

min_mse = mses[min_idx]

min_features = best_feature_sets[min_idx]

print('Minimum mse: ', min_mse)

print('Features in model: ', min_features)