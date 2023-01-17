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

import scipy

from matplotlib import pyplot as plt

#Задание 1

x = np.array([0.0, 0.4, 0.9, 1.6, 2.0])

def f(x):

    return 1/2 * (np.cos(x)**2).sum()

print(f(x))
#Задание 2

x = np.array([0.0, 0.4, 0.9, 1.6, 2.0])

w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

b = 3.8

def f2(x,w,b):

    return (x.dot(w)).sum() + b

print(f2(x,w,b))
#Задание 3

y = np.array([0.0, 0.4, 0.9, 1.6, 2.0])

t = np.array([1.0, 2.4, 2.9, 2.6, 3.0])

def loss(y,t):

    return ((y-t)**2).sum()

print(loss(y,t))
#Задание 4

from scipy import optimize

ff = np.array([1.0,5.5,16.0,

4.5,2.0,12.1,

8.0,1.0,13.5,

1.0,4.0,11.9,

5.0,5.5,20.9,

0.0,4.5,14.0,

1.0,6.0,17.4,

5.0,1.0,9.1,

2.0,4.5,17.3,

5.5,4.0,17.6,

6.5,0.0,11.0,

0.5,8.0,19.0,

3.0,7.0,21.0,

6.0,5.5,21.4,

1.5,2.5,10.4])

ff = ff.reshape(15,3)

w = ff[:,0]

x = ff[:,1]

y = ff[:,2]

b = 0.3

def loss(w):

    def f(w):

        return w.dot(x) +b

    t = f(w)

    return ((y-t)**2).sum()

r = optimize.minimize(loss,x0=w)

print(r.x)
#Задание 5

x = np.linspace(0.1,10,1000)

y = 1/x

plt.plot(x,y)

plt.grid()

plt.show()
#Задание 6

N = 1000

x = np.linspace(1,10,N)

def f(x,N):

    xi = []

    y = []

    for i in range(N):

        h = (x[-1]-1)/N

        xi.append(1+i*h)

        xi_ = np.array(xi)

        y.append(h*(1/xi_).sum())

    plt.plot(x,y)

    plt.grid()

    plt.show()

f(x,N)
#Задание 7

N = 100

sigm = 0.3

x = np.linspace(1,10,N)

def f(x):

    return np.sin(x)*(x/3)

y = f(x)

err = np.random.normal(loc=0.0,scale = sigm, size = N)

y_ = y+err

plt.plot(x,y)

plt.grid()

plt.scatter(x,y_,c='red')

plt.show()