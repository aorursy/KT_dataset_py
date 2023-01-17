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
import matplotlib.pyplot as plt

import numpy as np

X = 2 * np.random.rand(100, 1)

y = 4 + 3 * X + np.random.randn(100, 1)
plt.figure(figsize=(8, 5))

plt.plot(X, y, 'bo')

# plt.axis([140, 190, 45, 75])

plt.xlabel('X')

plt.ylabel('y')

plt.axis([0, 2, 0, 15])

plt.show()
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance 

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
X_new = np.array([[0], [2]])

X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance 

y_predict = X_new_b.dot(theta_best)

print(y_predict)
plt.figure(figsize=(8, 5))

plt.plot(X, y, 'bo')

# plt.axis([140, 190, 45, 75])

plt.xlabel('X')

plt.ylabel('y')

plt.plot(X_new, y_predict, "r-")

plt.plot(X, y, "b.")

plt.axis([0, 2, 0, 15])

plt.show()
from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()

lin_reg.fit(X, y)

print(lin_reg.intercept_, lin_reg.coef_)
lin_reg.predict(X_new) # Previous result: [[ 3.95064937][10.016376  ]]
eta = 0.1 # learning rate 

n_iterations = 1000 

m=100 # Number of instances

theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):

    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) 

    theta = theta - eta * gradients

print(theta)
theta_best  # Previous method result
print(lin_reg.intercept_, lin_reg.coef_)  # Previous method result
from pprint import pprint

import numpy as np
# Matrix

m1 = np.array([[ 23 , 402], 

               [69 , 221], 

               [118, 0]])

pprint(m1)

print("m1.shape =", m1.shape )

v1 = np.array ([149, 92, 313])

pprint(v1)
# Matrix adding

m1 = np.array([[23,402],

               [69,221],

               [118,0]])

m2 = np.array([[93,21],

               [223,11],

               [123,6]])

pprint(m1)

pprint(m2)

pprint((m1+m2))
pprint(3*m1)
# Matrix multiplication

m1 = np.array([[1,3,2],

               [4,0,1]])

m2 = np.array([[1,3],

               [0,1],

               [5,2]])

assert(m1.shape[1] == m2.shape[0])

m3 = np.dot(m1,m2)
pprint(m1)

pprint(m2)

pprint(m3)
# Determinant

m1 = np.array([[23,42,79], 

               [69,6,21], 

               [8,0,9]])

print(np.linalg.det(m1))

a = 23*6*9 + 42*21*8 - 79*6*8 - 42*69*9

print(a)
# Transpose

m1.T
#Inverse

# m1 = np.array([[0,5],

#                [.5,0]])

inv_m1 = np.linalg.inv(m1)

pprint(inv_m1)

pprint(np.dot(m1, inv_m1))
# Eigenvalues and eigenvectors

pprint(m1)

w,v = np.linalg.eig(m1)

print(w)

pprint(v)