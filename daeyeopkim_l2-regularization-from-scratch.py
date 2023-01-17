# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")

X = data.YearsExperience.values.reshape(1,-1)

Y = data.Salary.values.reshape(1,-1)

data.head()
X=(X-X.mean())/X.std()

Y=(Y-Y.mean())/Y.std()
plt.scatter(X, Y)

plt.show()
def initialize(unit=5):    

    

    

    W1 = np.random.randn(unit,1)

    b1 = np.zeros([unit,1])

    W2 = np.random.randn(1,unit)

    b2 = np.zeros([1,1])

    

    wb = {'W1':W1,'W2':W2,'b1':b1,'b2':b2}

    

    return wb
def forward_prop(wb,X):

    

    W1 = wb['W1']

    W2 = wb['W2']

    b1 = wb['b1']

    b2 = wb['b2']

    

    Z1 = np.dot(W1,X) + b1

    a1 = np.tanh(Z1)

    Z2 = np.dot(W2,a1) + b2

    a2 = Z2

    

    az = {'Z1':Z1, 'a1':a1, 'Z2':Z2, 'a2':a2}

    

    Y_hat = a2

    

    return Y_hat,az
def back_prop(Y_hat,Y,wb,az):

    

    m = Y.shape[1]

    

    W1 = wb['W1']

    W2 = wb['W2']

    b1 = wb['b1']

    b2 = wb['b2']

    

    Z1 = az['Z1']

    Z2 = az['Z2']

    a1 = az['a1']

    a2 = az['a2']



    dZ2 = Y_hat - Y

    dW2 = np.dot(dZ2, a1.T)/m

    db2 = np.sum(dZ2, axis=1, keepdims=True)/m

    da1 = np.dot(W2.T, dZ2)

    dZ1 = da1*a1*(1-a1)

    dW1 = np.dot(dZ1, X.T)/m

    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    

    dwb = {'dW2':dW2, 'dW1':dW1, 'db2':db2, 'db1':db1}

    

    return dwb



        
def update(wb,dwb,learning_rate=0.0001, lambd=0, m=1):

    

    dW1 = dwb['dW1']

    dW2 = dwb['dW2']

    db1 = dwb['db1']

    db2 = dwb['db2']

    

    W1 = wb['W1'] - learning_rate * (dW1 + lambd * wb['W1']/m)

    W2 = wb['W2'] - learning_rate * (dW2 + lambd * wb['W2']/m)

    b1 = wb['b1'] - learning_rate * db1

    b2 = wb['b2'] - learning_rate * db2

    

    wb = {'W1':W1,'W2':W2,'b1':b1,'b2':b2}

    

    

    return wb
def train(reg):

    wb = initialize(unit=80)

    for i in range(3000):

        Y_hat, az = forward_prop(wb, X)

        dwb = back_prop(Y_hat,Y,wb,az)

        wb = update(wb, dwb, learning_rate = 0.1, lambd = reg, m=X.shape[1])

    return wb
def train1(reg):

    wb = initialize(unit=30)

    for i in range(3000):

        Y_hat, az = forward_prop(wb, X)

        dwb = back_prop(Y_hat,Y,wb,az)

        wb = update(wb, dwb, learning_rate = 0.1, lambd = reg, m=X.shape[1])

    return wb
wb1 = train(0)

a1,_=forward_prop(wb1,X)



wb2 = train1(0.1)

a2,_=forward_prop(wb2,X)
f = plt.figure(figsize=(12,6))

ax1 = f.add_subplot(121)

ax2 = f.add_subplot(122)



ax1.plot(X.T,a1.T)

ax1.scatter(X.T,Y.T)

ax1.set_title('lambda = 0')



ax2.plot(X.T,a2.T)

ax2.scatter(X.T,Y.T)

ax2.set_title('lambda = 0.1')



plt.plot()
