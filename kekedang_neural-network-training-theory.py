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
import numpy as np



y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] #softmax function

t = [0, 0 , 1, 0, 0, 0, 0, 0, 0, 0] # one-hot encoding



def mean_squared_error(y, t):

    return 0.5 * np.sum((y-t)**2)



t = [0, 0 , 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(mean_squared_error(np.array(y), np.array(t)))



y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(mean_squared_error(np.array(y), np.array(t)))
def cross_entropy_error(y, t):

    delta = 1e-7 # Negative infinity prevention

    return -np.sum(t * np.log(y + delta))



t = [0, 0 , 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))



y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))
#mini-batch : The training is picked up only partially from the training data.

#Randomly selecting 100 out of 60,000 pieces of training data and using only those 100 pieces



train_size = 60000

batch_size = 10

batch_mask = np.random.choice(train_size, batch_size) # 10 random numbers from 0 to less than 60000

print(batch_mask) 
# cross_entropy_error(batch)



def cross_entropy_error(y, t):

    if y.ndim == 1:

        t.reshape(1, t.size)

        y.reshape(1, y.size)

        

    batch_size = y.shape[0]

    return -np.sum(t * np.log(y)) / batch_size
# cross_entrop_errer(no one-hot encoding)

"""

def cross_entropy_error(y, t):

    if y.ndim == 1:

        t.reshape(1, t.size)

        y.reshape(1, y.size)

        

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t])) / bath_size

"""
# bad



def numerical_diff(f, x):

    h = 10e-50 # rounding error

    return (f(x+h) - f(x)) / h # With errors
# good



def numerical_diff(f, x):

    h = 1e-4 # 0.0001

    return (f(x+h) - f(x-h)) / (2*h) # Central difference
def function_1(x):

    return 0.01*x**2 + 0.1*x
import numpy as np

import matplotlib.pylab as plt



x = np.arange(0.0, 15.0, 0.1)

y = function_1(x)

plt.xlabel("x")

plt.ylabel("f(x)")

plt.plot(x, y)

plt.show()
print(numerical_diff(function_1, 5))

print(numerical_diff(function_1, 7))
def function_2(x):

    return x[0]**2 + x[1]**2
def function_tmp1(x0):

    return x0*x0 + 4.0**2.0



numerical_diff(function_tmp1, 3.0)
def numerical_gradient_no_batch(f, x):

    h = 1e-4 # 0.0001

    grad = np.zeros_like(x)

    

    for idx in range(x.size):

        tmp_val = x[idx]

        

        # f(x+h)

        x[idx] = tmp_val + h

        fxh1 = f(x)

        

        # f(x-h)

        x[idx] = tmp_val - h

        fxh2 = f(x)

        

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val

        

    return grad
print(numerical_gradient_no_batch(function_2, np.array([3.0, 4.0])))

print(numerical_gradient_no_batch(function_2, np.array([0.0, 5.0])))

print(numerical_gradient_no_batch(function_2, np.array([6.0, 2.0])))
def numerical_gradient(f, X):

    if X.ndim == 1:

        return numerical_gradient_no_batch(f, X)

    else:

        grad = np.zeros_like(X)

        

        for idx, x in enumerate(X):

            grad[idx] = numerical_gradient_no_batch(f, x)

        

        return grad
# f : Function to optimize

# init_x : Initial value

# lr : learning rate

# Number of iterations according to the gradient method



def gradient_descent(f, init_x, lr=0.01, step_num=100):

    x = init_x

    

    for i in range(step_num):

        grad = numerical_gradient(f, x)

        x -= lr * grad

    return x
def function_2(x):

    return x[0]**2 + x[1]**2



init_x = np.array([-3.0, 4.0])

gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# Example learning rate is too high

init_x = np.array([-3.0, 4.0])

print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))



# Example learning rate is too small

init_x = np.array([-3.0, 4.0])

print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
def softmax(a):

    c = np.max(a)

    exp_a = np.exp(a - c)

    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a

    

    return y

    

class simpleNet:

    def __init__(self):

        self.W = np.random.randn(2,3) 

        

    def predict(self, x):

        return np.dot(x, self.W)



    def loss(self, x, t):

        z = self.predict(x)

        y = softmax(z)

        loss = cross_entropy_error(y, t)



        return loss
net = simpleNet()

print(net.W)



x = np.array([0.6, 0.9])

p = net.predict(x)

print(p)



print(np.argmax(p))



t = np.array([0, 0, 1])

print(net.loss(x, t))
def f(W):

    return net.loss(x, t)



dW = numerical_gradient(f, net.W)

print(dW)



# lambda 

f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)