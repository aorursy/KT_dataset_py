# Import library

import math

import time

import numpy as np
# Sigmoid



def sigmoid(x):

    s = 1/(1+np.exp(-x))

    return s

x = np.array([1, 2, 3])

sigmoid(x)
# Sigmoid gradient

# sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))



def sigmoid_d(x):

    s = sigmoid(x)

    ds = s*(1-s)

    return ds

x = np.array([1, 2, 3])

print ("sigmoid_derivative(x) = " + str(sigmoid_d(x)))
# Reshaping image array

# image array shape (length,height, depth= 3 typical RGB value) --> (length*height*depth , 1)



def image2vector(image):

    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))

    return v

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
# Normalizing rows



def norm(x):

    x_norm = np.linalg.norm(x,axis=1,keepdims=True)

    x = x/x_norm

    return x

x = np.array([

    [0, 3, 4],

    [1, 6, 4]])

print("normalizeRows(x) = " + str(norm(x)))
# Softmax; use when there are more than two or more classes

# if matrix is m x n 



def softmax(x):

    s = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True) # exp(x) / sum(exp(x)) by rows

    return s

x = np.array([

    [9, 2, 5, 0, 0],

    [7, 5, 0, 0 ,0]])

print("softmax(x) = " + str(softmax(x)))
# Vectorization

# example: x1*W'



x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]

x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

W = np.random.rand(3,len(x1))

tic = time.process_time()

dot = np.dot(W,x1)

toc = time.process_time()

print("Computation time: " + str(1000*(toc - tic)) + "ms\n"  + str(dot))
# L1 loss function



def L1(yhat,y):

    loss = np.sum(np.abs(yhat-y))

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

print("L1 = " + str(L1(yhat,y)))
# L2 loss funciton



def L2(yhat,y):    

    loss = np.sum((yhat-y)**2)

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2(yhat,y)))