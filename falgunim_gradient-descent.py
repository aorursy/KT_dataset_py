# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train = train.dropna(how = "any")
train.head()
print(train.shape)
print("------------")
print(train.columns)
print("------------")
print(train.dtypes)
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(train.x,train.y,marker=".")
#data = train.as_matrix()
data = train.values
#hypothesis function, y_hat = mx+b
def y_hat(x, m, b):
    return m*x + b

#Sum of Square Error
def SSE(m,b,data):
    err = 0
    for i in range(len(data)):
        x = data[i,0]
        y = data[i,1]
        err += ( y - y_hat(x, m, b))**2
        #print(y, err)
    return err

#Mean Square Error
def MSE(m,b,data):
    return SSE(m,b,data)/len(data)
# Testing with y_hat = 2x+3  | We assumed random numbers m=2 and b=3
print("SSE =", SSE(2,3,data))
print("MSE =", MSE(2,3,data))
plt.scatter(train.x.values,train.y.values)
plt.plot(train.x.values, 2*train.x.values + 3, color ="red")
#vanila gradient descent / a.k.a. batch gradient descent
def gradient_descent(m,b,data,learning_rate):
    m_gradient = 0
    b_gradient = 0
    N=len(data)
    
    for i in range(N):
        x = data[i,0]
        y = data[i,1]
        m_gradient += -(2/N) * (x) * (y - y_hat(x, m, b))
        b_gradient += -(2/N) * (y - y_hat(x, m, b))
        
    new_m = m - (m_gradient*learning_rate)
    new_b = b - (b_gradient*learning_rate)
    
    return new_m,new_b    
    
#Single step of gradient descent
a,b = gradient_descent(2,3,data,0.0001)
MSE(a,b,data)
# Now the whole - taking 10000 iteration
m=2
b=3
for i in range(10000):
    m,b = gradient_descent(m,b,data,0.0001)
print("final values for 10k repetitions")
print("%.2f   %.2f   %.2f" %(m,b,MSE(m,b,data)))
    
plt.figure(figsize=(10,8))
plt.scatter(train.x.values,train.y.values,marker=".")
plt.plot(train.x.values, (2*train.x.values)+3,color ="red",linewidth=7)
plt.plot(train.x.values,(0.97*train.x.values)+1.77, color ="g",linewidth=7)