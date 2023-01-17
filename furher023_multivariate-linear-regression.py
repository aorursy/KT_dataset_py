# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/housedata/data.csv')
data.columns
#Eliminating the columnns which are not useful for prediction as of now
data = data.drop(columns=['country','statezip','city','street','date'])
data.columns
data.tail(7)
# Some operations on data so that gradient descent will converge faster
data = data-data.mean()
data
data=data/data.std()
data
X=data[data.columns[2:]].to_numpy()
type(X)
a=np.ones((4600,1) )
type(a)
X=np.append(a,X,axis=1)
X
y=data['price'].to_numpy()
y= np.reshape(y,(len(y),1))
y
X.shape

# returns the value of cost function corresponding to parameters theta at given point
def cost(th,m):
    h = np.matmul(X,th)
    h = h - y
    h= np.sum(np.square(h))/2*m
    return h
    
cost(theta,4600)
def gradient_descent(alpha):
    theta=np.zeros((12,1))
    theta.shape
    J=[]
    it=[]
    m=4600
    #print(theta.shape)
    for i in range(1,100001):
        h = np.matmul(X,theta)
        #print(h.shape)
        h = h - y
        #print(h.shape)
        #print(X.shape)
        h = np.sum(np.multiply(h,X),axis=0)*(alpha/m)
        theta = theta - h
        J.append(cost(theta,m))
        it.append(i)
    print('complete')
    return(J,it,theta)
# Plotting the cost function vs no. of iterations for different values of alpha
%matplotlib inline
from matplotlib import pyplot as plt
(J,it,theta)=gradient_descent(0.0001)
plt.plot(it,J)
plt.xlabel('Iterations ->')
plt.ylabel('Cost function ->')



(J,it,theta)=gradient_descent(0.0003)
plt.plot(it,J)
plt.xlabel('Iterations ->')
plt.ylabel('Cost function ->')
(J,it,theta)=gradient_descent(0.00001)
plt.plot(it,J)
plt.xlabel('Iterations ->')
plt.ylabel('Cost function ->')
(J,it,theta)=gradient_descent(0.00003)
plt.plot(it,J)
plt.xlabel('Iterations ->')
plt.ylabel('Cost function ->')
(J,it,theta)=gradient_descent(0.0001)
plt.plot(it,J)
plt.xlabel('Iterations ->')
plt.ylabel('Cost function ->')
