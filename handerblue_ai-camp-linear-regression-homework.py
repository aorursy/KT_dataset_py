import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from sklearn.linear_model import LinearRegression # sk-learn库Linear Regression模型

from sklearn.model_selection import train_test_split # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库

import math #数学库
data = pd.read_csv("../input/kc_house_data.csv")
data.dtypes
X = data[['bedrooms','bathrooms','sqft_living','floors']]

Y = data['price']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 1/3, random_state = 0)
plt.scatter(data['sqft_living'], Y)

plt.show()
plt.hist(data['sqft_living'])

plt.show()
xtrain = np.asmatrix(xtrain)

xtest = np.asmatrix(xtest)

ytrain = np.ravel(ytrain)

ytest = np.ravel(ytest)



model = LinearRegression()

model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model.intercept_
#一个房子，3个卧室，2个卫生间，2500sqft，2层楼，预测房价

model.predict([[3,2,2500,2]])
pred = model.predict(xtest)

((pred-ytest)**2).sum() / len(ytest)
(abs(pred-ytest)/ytest).sum() / len(ytest)
predtest = model.predict(X)

((predtest-Y)*(predtest-Y)).sum() / len(Y)
(abs(predtest-Y)/Y).sum() / len(Y)
All_features = ['bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']



# Given some options for feature slection

features = [['sqft_living','sqft_lot', 'sqft_lot15', 'sqft_above', 'sqft_living15'],

            ['bedrooms','bathrooms','sqft_living','floors', 'yr_built', 'sqft_above', 'view'], 

            ['bedrooms','bathrooms','grade', 'sqft_living','floors', 'yr_built', 'sqft_above', 'view'],

            ['sqft_living','sqft_lot', 'waterfront','view','floors', 'yr_built', 'sqft_above', 'sqft_lot15', 'sqft_basement'],

            All_features]
from sklearn.model_selection import cross_val_score



mses, e_vars = [], []

for i, feature in enumerate(features):

    X = data[feature]

    Y = data["price"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

    model = LinearRegression()

    model.fit(X_train, Y_train)

    prediction = model.predict(X_test)

    

    # Cross validation

    cross_val = cross_val_score(model, X_test, Y_test, cv=10, scoring="explained_variance")

    e_var = cross_val.sum() / len(cross_val)

    e_vars.append(e_var)

    

    #MSE

    MSE = ((prediction - Y_test)**2).sum() /len(Y_test)

    mses.append(MSE)

    

    print("\nfeature select " + str(i), "MSE =", MSE)

    print("Avg explained_variance:", e_var)

    

best_e_vars_idx = np.array(e_vars).argmax()

best_feature_idx = np.array(mses).argmin()

if best_e_vars_idx != best_feature_idx:

    print("TBD: Would evaluation results of MSE and Varaiance be different?")
best_feature = features[best_e_vars_idx]

print("Best feature set =", best_feature)
data_X = data[best_feature]

data_Y = data["price"]

model = LinearRegression()

model.fit(data_X, data_Y)

prediction = model.predict(data_X)

MSE = ((prediction - data_Y)**2).sum() / len(data_Y)
print("Using features:\n", best_feature)

print("MSE =",MSE)



pd.DataFrame(list(zip(data_X.columns, np.transpose(model.coef_))))