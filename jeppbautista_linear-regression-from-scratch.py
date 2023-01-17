# import python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import time
def linear(X, b0, b1):
    return [b0+b1*x for x in X]
# b0 - Intercept
def intercept(X, Y, b1): 
    x_ = np.mean(X)
    y_ = np.mean(Y)
    
    return y_-b1*x_
# b1 - Slope
def slope(X, Y):
    x_ = np.mean(X)
    y_ = np.mean(Y)
    
    rise = sum([(x-x_) * (y-y_) for x,y in zip(X,Y)])
    run = sum([(x-x_)**2 for x,y in zip(X,Y)])
    
    return rise / run
data = pd.read_csv("../input/Automobile_data.csv")
data.head()
print("Dataset size")
print("Rows {} Columns {}".format(data.shape[0], data.shape[1]))
print("Columns and data types")
pd.DataFrame(data.dtypes).rename(columns = {0:'dtype'})
try:
    data[['price']] = data[['price']].astype(int)
except ValueError:
    print("Trying out the line of code above will result to this error:\n")
    print("Value Error: invalid literal for int() with base 10: '?'")
data['price'].value_counts()[:5]
data = data.loc[data['price']!='?']
data[['price']] = data[['price']].astype(int)
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True)
plt.show()
final_data = data[['engine-size', 'price']]
predictor = data['engine-size']
target = data['price']
plt.figure(figsize=(8,5))
plt.title("Price vs engine-size")
plt.scatter(predictor, target, color = "#247ba0")
plt.xlabel('engine-size')
plt.ylabel('price')
plt.show()
b1 = slope(predictor, target)
b0 = intercept(predictor, target, b1)
predicted = linear(predictor, b0, b1)
# print(predicted) - Uncomment to see predicted values
plt.figure(figsize = (8, 5))
plt.plot(predictor, predicted, color = '#f25f5c')
plt.scatter(predictor, predicted, color = '#f25f5c')
plt.title('Predicted values by Linear Regression', fontsize = 15)
plt.xlabel('engine-size')
plt.ylabel('price')
plt.scatter(predictor, target, color = "#247ba0")
plt.show()
print("Coefficients:\n=============")
print("b0 : ", b0)
print("b1 : ", b1)
def r_squared(Y, Y_HAT):
    ssr, sse, r_sqr = [0]*3
    y_ = np.mean(Y)
#     ssr = sum([(y_hat - y_)**2 for y_hat in Y_HAT])
    sse = sum([(y - y_hat)**2 for y,y_hat in zip(Y, Y_HAT)])
    sst = sum([(y - y_)**2 for y in Y])
    
    r_sqr = 1 - (sse / sst)
    
    return r_sqr
    
r_squared(target, predicted)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression()
predictor = np.array(predictor).reshape((-1, 1))
reg = reg.fit(predictor, target)

Y_pred = reg.predict(predictor)
r2_score = reg.score(predictor, target)
print(r2_score)