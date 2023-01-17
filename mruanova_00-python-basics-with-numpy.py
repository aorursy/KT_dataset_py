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
import math

def basic_sigmoid(x):

    s = 1 / (1 + math.exp(-x))

    return s

basic_sigmoid(3)
import numpy as np

x = np.array([1, 2, 3])

print(np.exp(x)) # result is (exp(1), exp(2), exp(3))
# example of vector operation

x = np.array([1, 2, 3])

print (x + 3)
def sigmoid(x):

    s = 1 / (1 + np.exp(-x))    

    return s

x = np.array([1, 2, 3])

sigmoid(x)
def sigmoid_derivative(x):

    s = 1.0 / (1 + np.exp(-x))

    ds = s * (1.0 - s)

    return ds

x = np.array([1, 2, 3])

print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
def image2vector(image):

    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

    return v

import numpy as np 

image = np.array([[[ 0.67826139,  0.29380381],

        [ 0.90714982,  0.52835647],

        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],

        [ 0.85304703,  0.52351845],

        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],

        [ 0.10820313,  0.49978937],

        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))
def normalizeRows(x):

    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)

    x = x / x_norm

    return x

x = np.array([

    [0, 3, 4],

    [1, 6, 4]])

print("normalizeRows(x) = " + str(normalizeRows(x)))
def softmax(x):

    x_exp = np.exp(x)

    x_sum = np.sum(x_exp, axis = 1, keepdims = True)

    s = x_exp / x_sum

    return s

x = np.array([

    [9, 2, 5, 0, 0],

    [7, 5, 0, 0 ,0]])

print("softmax(x) = " + str(softmax(x)))
# vectorization

import time



x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]

x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]



### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###

tic = time.process_time()

dot = 0

for i in range(len(x1)):

    dot+= x1[i]*x2[i]

toc = time.process_time()

print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")



### CLASSIC OUTER PRODUCT IMPLEMENTATION ###

tic = time.process_time()

outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros

for i in range(len(x1)):

    for j in range(len(x2)):

        outer[i,j] = x1[i]*x2[j]

toc = time.process_time()

print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")



### CLASSIC ELEMENTWISE IMPLEMENTATION ###

tic = time.process_time()

mul = np.zeros(len(x1))

for i in range(len(x1)):

    mul[i] = x1[i]*x2[i]

toc = time.process_time()

print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")



### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###

W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array

tic = time.process_time()

gdot = np.zeros(W.shape[0])

for i in range(W.shape[0]):

    for j in range(len(x1)):

        gdot[i] += W[i,j]*x1[j]

toc = time.process_time()

print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
# without looping

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]

x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]



### VECTORIZED DOT PRODUCT OF VECTORS ###

tic = time.process_time()

dot = np.dot(x1,x2)

toc = time.process_time()

print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")



### VECTORIZED OUTER PRODUCT ###

tic = time.process_time()

outer = np.outer(x1,x2)

toc = time.process_time()

print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")



### VECTORIZED ELEMENTWISE MULTIPLICATION ###

tic = time.process_time()

mul = np.multiply(x1,x2)

toc = time.process_time()

print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")



### VECTORIZED GENERAL DOT PRODUCT ###

tic = time.process_time()

dot = np.dot(W,x1)

toc = time.process_time()

print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
def L1(yhat, y):

    loss = np.sum(np.absolute(yhat - y))

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

print("L1 = " + str(L1(yhat,y)))
def L2(yhat, y):

    loss = np.dot(y-yhat, y-yhat)

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2(yhat,y)))