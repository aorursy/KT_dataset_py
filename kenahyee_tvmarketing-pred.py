import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
tv = pd.read_csv('/kaggle/input/tvmarketing-dataset/tvmarketing.csv')
tv
plt.scatter(tv['TV'], tv['Sales'], color ='green')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
msk = np.random.rand(len(tv))<0.8
train = tv[msk]
test = tv[~msk]
train.shape
test.shape
plt.scatter(train['TV'], train['Sales'], color ='darkblue')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
from sklearn import linear_model
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['TV']])
y_train = np.asanyarray(train[['Sales']])
regr.fit (x_train, y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter(train.TV, train.Sales,  color='blue')
plt.plot(x_train, regr.coef_[0][0]*x_train + regr.intercept_[0], '-r')
plt.xlabel("TV")
plt.ylabel("Sales")
regr.score(x_train, y_train)
from sklearn.metrics import r2_score

x_test = np.asanyarray(test[['TV']])
y_test = np.asanyarray(test[['Sales']])
y_test_pred = regr.predict(x_test)

y_test_pred
