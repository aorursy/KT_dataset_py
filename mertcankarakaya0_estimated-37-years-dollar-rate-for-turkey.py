# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/37-years-dollar-rate-for-turkey/37.year.dollar.rate.for.Turkey.csv",sep=';')
x = data.Day
y = data.Rate_Price
x = x.values.reshape(-1,1)
y= y.values.reshape(-1,1)
import matplotlib.pyplot as plt
plt.scatter(x,y,c="gold",s=3) 
plt.xlabel("Rate_Price")
plt.ylabel("Day")
plt.show()
from sklearn.linear_model import LinearRegression
estimateLinear = LinearRegression()
estimateLinear.fit(x,y)
y_head_linear=estimateLinear.predict(x)
plt.plot(x,y_head_linear,c="red")
plt.title("Linear Regression")
plt.show()
MSE_Linear = 0
for i in range(len(y)):
    MSE_Linear = MSE_Linear + (float(y[i])-float(y_head_linear[i]))**2
mse_linear_value=MSE_Linear/len(y)
print("MSE value of Linear Regression : ",mse_linear_value)

from sklearn.preprocessing import PolynomialFeatures
estimatePolynomial = PolynomialFeatures(degree=3)
X_new = estimatePolynomial.fit_transform(x)
polynomial_model = LinearRegression()
polynomial_model.fit(X_new,y)
y_head_polynomial=polynomial_model.predict(X_new)
plt.plot(x,y_head_polynomial,c="green")
plt.title("Polynomial Regression")
plt.show()
MSE_polynomial = 0
for i in range(len(X_new)):
    MSE_polynomial = MSE_polynomial + (float(y[i])-float(y_head_polynomial[i]))**2
mse_polynomial_value=MSE_polynomial/len(X_new)
print("3th degrees MSE value of Polynomial Regression ",mse_polynomial_value)
mse_array=[]
MSE_polynomial = 0
for a in range(75):
    estimate_polynomial = PolynomialFeatures(degree=a+1)
    X_new = estimate_polynomial.fit_transform(x)
    polynomial_model = LinearRegression()
    polynomial_model.fit(X_new,y)
    y_head_polynomial2=polynomial_model.predict(X_new)
    for i in range(len(X_new)):
        MSE_polynomial = MSE_polynomial + (float(y[i])-float(y_head_polynomial2[i]))**2
    mse_polynomial_value=MSE_polynomial/len(X_new)
    mse_array.append(mse_polynomial_value)
    print(a+1,".degrees error in function:", mse_polynomial_value)
    MSE_polynomial = 0
min_value=0
for value in range(75):
    if(mse_array[value]==min(mse_array)):
        print("Minimum value:",mse_array[value]," Degree:",(value+1))
        min_value=value+1
estimate_polynomial_min = PolynomialFeatures(degree=min_value)
X_new = estimate_polynomial_min.fit_transform(x)
polynomial_model_min = LinearRegression()
polynomial_model_min.fit(X_new,y)
y_head_polynomial_min=polynomial_model_min.predict(X_new)
plt.plot(x,y_head_polynomial_min,c="blue")
plt.title("Polynomial Regression (the best degree)")
plt.show()
plt.plot(x,y_head_linear,c="red")
plt.plot(x,y_head_polynomial,c="green")
plt.plot(x,y_head_polynomial_min,c="blue")
plt.show()
fig, axs = plt.subplots(2, 2,figsize=(20,10))
#Table-1    
axs[0, 0].plot(x,y_head_linear,c="red")
axs[0, 0].set_title('Linear Regression')
axs[0, 0].scatter(x,y,c="gold",s=3)
#Table-2
axs[0, 1].plot(x,y_head_polynomial,c="green")
axs[0, 1].set_title('3th degree Polynomial Regression')
axs[0, 1].scatter(x,y,c="gold",s=3)
#Table-3
axs[1, 0].plot(x,y_head_polynomial_min,c="blue")
axs[1, 0].set_title('Polynomial Regression of the best degree(4)')
axs[1, 0].scatter(x,y,c="gold",s=3)
#Table-4 (In the last table, we do it to draw all of them.)
axs[1, 1].plot(x,y_head_linear,c="red")
axs[1, 1].plot(x,y_head_polynomial,c="green")
axs[1, 1].plot(x,y_head_polynomial_min,c="blue")
axs[1, 1].set_title('Combination of 3 Tables')
axs[1, 1].scatter(x,y,c="gold",s=3)
plt.show()

#I entered the axis labels of the tables.
for ax in axs.flat:
    ax.set(xlabel='Day', ylabel='Rate Price')

#I hid the X tags and were able to see the tags for the top graphics
for ax in axs.flat:
    ax.label_outer()

print("Real Value:",y[520]," Estimated Value:",y_head_polynomial_min[520])
print("Difference between real value and estimated value:",(float(y[520])-float(y_head_polynomial_min[520])))
