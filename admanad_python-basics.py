import math

import numpy as np



def basic_func(x):

    s = 1/x

    return s
x = np.array([1,2,3])
theta = np.array([2,2,2])
np.dot(x,theta)
X = np.array([[1,2,3], [1,2,3]])

X
np.dot(X, theta)
X@theta


# 1D array 

vector_a = np.array([[1, 4], [5, 6]]) 

vector_b = np.array([2, 4]) 

  

product = np.dot(vector_a, vector_b) 

print("Dot Product  : \n", product) 
np.eye(3)
o = np.ones((2,2))
o.reshape((4,1))
o.reshape((1,4))
o.reshape((-1,4))
o.reshape((4,-1))