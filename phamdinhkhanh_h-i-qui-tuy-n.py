import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data 
plt.plot(X, y, 'r-^')
plt.axis([145, 185, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('height vs weight')
# plt.show()
w1 = (np.mean(X)*np.mean(y)-np.mean(X*y))/((np.mean(X))**2-(np.mean(X**2)))
w0 = np.mean(y)-w1*np.mean(X)

print('w1: ', w1)
print('w0: ', w0)
one = np.ones(X.shape[0]).reshape(-1, 1)
Xbar = np.concatenate([one, X], axis = 1)
Xbar
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
w
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=True) # fit_intercept = False for calculating the bias
regr.fit(X, y)

# Compare two results
print( 'Coefficient : ', regr.coef_ )
print( 'Interception  : ', regr.intercept_ )
regr.predict(np.array([[120]]))
# print(X.shape)
# import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

X = np.arange(10)
noise = np.random.randn(10)
y = 2*X + noise
X = X.reshape(-1, 1)


plt.plot(X, y, 'ro-', label = 'line 1', linewidth = 2)

print('X shape: ', X.shape)
print('y shape: ', y.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept = True)
model.fit(X.reshape(-1, 1), y)
model
print('Model Intercept: {}, Coeficients: {}'.format(model.intercept_,model.coef_))
# Predict
yhat = model.predict(X)
yhat
# Visualization
plt.plot(X, y, label = 'Actual')
plt.plot(X, yhat,'r--', label = 'Predicted')
plt.legend()
# Hệ số R^2
model.score(X, y)
# Kiểm tra R^2 theo công thức 1-ESS/TSS
def _ESS(y, yhat):
    return np.sum((y-yhat)**2)

def _RSS(y, yhat):
    ybar = np.mean(yhat)
    return np.sum((yhat-ybar)**2)

def _TSS(y, yhat):
    ybar = np.mean(yhat)
    return np.sum((y-ybar)**2)
    
TSS = _TSS(y, yhat)
ESS = _ESS(y, yhat)
print('R squared: {:.8f}'.format(1-ESS/TSS))
ones = np.ones((X.shape[0], 1))
Xbar = np.concatenate((ones, X), axis = 1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

w = np.dot(np.linalg.inv(A), b)
print('Estimate coeficients: ', str(w))
from sklearn.linear_model import Ridge

rid_mod = Ridge(alpha = 0.3, normalize = True)
rid_mod.fit(X, y)

yhat_rid = rid_mod.predict(X)

plt.plot(X, y, label = 'Actual')
plt.plot(X, yhat_rid,'r--', label = 'Predicted')
plt.legend()
# R^2
rid_mod.score(X, y)

# \\TO DO: Tìm hiều phương pháp lựa chọn hệ số alpha tối ưu thông qua RidgeCV và AlphaSelection
from sklearn.linear_model import Lasso

las_mod = Lasso(alpha = 0.1, normalize = True)
las_mod.fit(X, y)

yhat_las = las_mod.predict(X)

plt.plot(X, y, label = 'Actual')
plt.plot(X, yhat_las,'r--', label = 'Predicted')
plt.legend()
las_mod.score(X, y)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('https://raw.githubusercontent.com/phamdinhkhanh/LSTM/master/international-airline-passengers.csv', sep = ',', 
                   header = 0, 
                   index_col = 0)

data.head()
data.columns = ['Y']
data.head()
data.plot()
X1 = data.shift(1)
X2 = data.shift(2)
X2.head()
X1 = X1.fillna(X1.iloc[1, 0]-(X1.iloc[2, 0]-X1.iloc[1, 0]))
# X2 = X2.fillna(X1.iloc[1, 0]-(X1.iloc[2, 0]-X1.iloc[1, 0]))
delta = X1.iloc[3, 0]-X1.iloc[2, 0]
X2.iloc[1, 0] = X2.iloc[2, 0] - delta
X2.iloc[0, 0] = X2.iloc[1, 0] - delta
X2.head()
# Tao dataframe X
X = pd.concat((X1, X2), axis = 1)
X.iloc[0, 0] = 98
X.head()
y = data
X_train, y_train = X.iloc[:-12,:], y.iloc[:-12,]
X_test, y_test = X.iloc[-12:, :], y.iloc[-12:,]
y_train.plot()
y_test.plot()
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept = True)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
y_train.plot()
plt.plot(y_pred)
y_pred_test = model.predict(X_test)
y_test.plot()
plt.plot(y_pred_test)
# danh gia mo hinh tren RMSE
from math import sqrt
def rmse(y, yhat):
    return sqrt(np.mean((y-yhat)**2))

rmse(y_test, y_pred_test)
rmse(y_train, y_pred)