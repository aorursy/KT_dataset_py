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
import matplotlib.pyplot as plt
%matplotlib inline
train.shape
train.columns
plt.scatter(train.x,train.y,marker=".")
data = train.as_matrix()
def SSE(m,b,data):
    err = 0
    for i in range(len(data)):
        x = data[i,0]
        y = data[i,1]
        y_hat = m*x + b
        err += (y-y_hat)**2
        #print(y,    y_hat,    err)
    return err

SSE(2,3,data)
plt.scatter(train.x.values,train.y.values)
plt.plot(train.x.values, 2*train.x.values + 3, color ="red")
def gradient_descent(m,b,data,learning_rate):
    m_gradient = 0
    b_gradient = 0
    N=len(data)
    for i in range(N):
        x = data[i,0]
        y = data[i,1]
        m_gradient += -(2/N)*(x)*(y-(m*x+b))
        b_gradient += -(2/N)*(y-(m*x+b))
    new_m = m - (m_gradient*learning_rate)
    new_b = b - (b_gradient*learning_rate)
    return new_m,new_b    
    
a,b = gradient_descent(2,3, data,0.0001)
SSE(2,3,data)
SSE(a,b,data)
m=2
b=3
for i in range(10000):
    m,b = gradient_descent(m,b,data,0.0001)
print("final values for 10k repetitions")
print("%.2f   %.2f   %.2f" %(m,b,SSE(m,b,data)))
    
plt.figure(figsize=(10,8))
plt.scatter(train.x.values,train.y.values,marker=".")
plt.plot(train.x.values, (2*train.x.values)+3,color ="red",linewidth=7)
plt.plot(train.x.values,(0.97*train.x.values)+1.77, color ="g",linewidth=7)
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

def error(x, y):
    return SSE(x,y,data)

m = np.arange(1,2,0.01)
b = np.arange(-0.5,0.5,0.01)


fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121, projection='3d')
ax.view_init(elev=20.0, azim=115)

X, Y = np.meshgrid(m, b)

zs = np.array([error(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z,cmap='hot')

ax.set_title('Gradient Descent')
ax.set_xlabel('slope (m)')
ax.set_ylabel('y-intercept (b)')
ax.set_zlabel('Error')

#PLOT2
ax2 = fig.add_subplot(122, projection='3d')
ax2.view_init(elev=50.0, azim=150)

X, Y = np.meshgrid(m, b)

zs = np.array([error(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax2.plot_surface(X, Y, Z,cmap='hot')

ax2.set_title('Gradient Descent')
ax2.set_xlabel('slope (m)')
ax2.set_ylabel('y-intercept (b)')
ax2.set_zlabel('Error')


plt.show()

fig = plt.figure(figsize=(20,7))
ax = fig.add_subplot(121, projection='3d')
ax.view_init(elev=20.0, azim=115)

X, Y = np.meshgrid(m, b)
ax.plot_surface(X, Y, Z,cmap='hot')
zs = np.array([error(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
X
