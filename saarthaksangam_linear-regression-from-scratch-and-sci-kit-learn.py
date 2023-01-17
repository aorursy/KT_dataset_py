%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading csv
data = pd.read_csv('../input/headbrain.csv')
print(data.shape)
data.head()

# extracting the column values
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# calculating mean
mean_x = np.mean(X)
mean_y = np.mean(Y)
m = len(X)
# check formula to find the coefficient b0 and b1 and you will understand the code
num = 0
den = 0
for i in range(m):
    num += (X[i] - mean_x) * (Y[i] - mean_y)
    den += (X[i] - mean_x) ** 2

b1 = num / den
b0 = mean_y - (b1 * mean_x)

print(b1, b0)
#plotting values 
x_max = np.max(X) + 100
x_min = np.min(X) - 100
#calculating line values of x and y
x = np.linspace(x_min, x_max, 1000)
y = b0 + b1 * x
#plotting line 
plt.plot(x, y, color='#00ff00', label='Linear Regression')
#plot the data point
plt.scatter(X, Y, color='#ff0000', label='Data Point')
# x-axis label
plt.xlabel('Head Size (cm^3)')
#y-axis label
plt.ylabel('Brain Weight (grams)')
plt.legend()
plt.show()
# calculating root mean squared error - again check the formula
rmse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print("RMSE Score = ", float(rmse))
# calculating r-squared score (check formula for clarity)
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2

r2 = 1 - (ss_r / ss_t)
print("R^2 Score = ",  float(r2))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Can not use rank 1 matrix in Sklearn
X = X.reshape((m,1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)
# Calculating RMSE and R^2 Score
# mse = mean_squared_error(Y, Y_pred)
# rmse = np.sqrt(mse)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2_score = reg.score(X, Y)

print("RMSE = {} \t R^2 Score = {}".format(rmse, r2_score))
#print(r2_score)