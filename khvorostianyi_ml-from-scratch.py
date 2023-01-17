import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import requests
from IPython.display import display, Math, Latex
df = pd.read_csv("../input/headbrain.csv")
df.head()
X = df.iloc[:,-2].values
Y = df.iloc[:,-1].values
print(X.shape)
print(Y.shape)
def ord_LinReg_fit(X,Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    m = len(X)

    number = 0
    denom  = 0
    for i in range(m):
        number += (X[i] - x_mean)*(Y[i] - y_mean)
        denom += (X[i] - x_mean)**2

    # coef.
    b_1 = number/denom
    b_0 = y_mean - (b_1*x_mean)
    return b_0,b_1
max_x = np.max(X) + 100
min_x = np.min(X) - 100


x = np.linspace(min_x, max_x, 1000)
b = ord_LinReg_fit(X,Y)
y = b[0] + b[1] * x
plt.plot(x,y, color='g')
plt.scatter(X, Y, c='b', label='Scatter Plot')
y_pred = [(lambda x:b[0] + b[1]*x)(x) for x in X]
def rmse(Y, y_pred):
    rmse = 0
    m = len(Y)
    for i in range(m):
        rmse += (Y[i] - y_pred[i])**2
    rmse = np.sqrt(rmse/m)
    return rmse
rmse(Y,y_pred)
Y[0],y_pred[0]
def r_2(Y,y_pred):
    ss_t = 0
    ss_r = 0
    y_mean = np.mean(Y)
    for i in range(len(Y)):
        ss_t += (Y[i] - y_mean)**2
        ss_r += (Y[i] - y_pred[i])**2
    r_2 = 1 - (ss_r/ss_t)
    return r_2
r_2(Y,y_pred)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((len(df), 1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Calculating RMSE and R2 Score
mse = mean_squared_error(Y, Y_pred)
r2_score = reg.score(X, Y)

print(np.sqrt(mse))
print(r2_score)
def get_X(X):
    m = len(X)
    X = np.c_[np.ones(m),X]
    return X
def LSM_fit(X,Y):
    X_inv = np.linalg.inv(np.matmul(X.T,X))
    middle_res = np.matmul(X_inv,X.T)
    theta = np.matmul(middle_res,Y)
    return theta
X_ = get_X(df.iloc[:,:3].values)
theta = LSM_fit(X_, Y)
print(X_.shape)
def LSM_grad_predict(X,theta):
    result = []
    for i in X:       
        r = sum(theta*i)
        result.append(r)
    return result
y_pred_lsm = LSM_grad_predict(X_, theta)
Y[0], y_pred_lsm[0]
rmse(Y,y_pred_lsm)
r_2(Y,y_pred_lsm)
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J
B = [0,0,0,0]
inital_cost = cost_function(X_, Y, B)
print(inital_cost)
def gradient_descent(X, Y, B, alpha, iterations):
    
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history
B = [0,0,0,0]
alpha = .0000001
newB, hist = gradient_descent(X_, Y, B, alpha, 1000)
newB
y_pred_gd = LSM_grad_predict(X_,newB)
print(f'{Y[0]} =>{y_pred_gd[0]}')
rmse(Y,y_pred_gd)
r_2(Y,y_pred_gd)