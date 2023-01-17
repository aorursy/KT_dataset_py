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
import numpy as np

import random

input = np.linspace(-10,10,100)



def sigmoid(x):

    return 1/(1+np.exp(-x))



def sigmoid_der(x):

    return sigmoid(x)*(1-sigmoid(x))



#plot sigmoid

#from matplotlib import pyplot as plt

#plt.plot(input,sigmoid(input),c="r")



feature_set=np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])

labels=np.array([[1,0,0,1,1]])

labels=labels.reshape(5,1)

np.random.seed(42)

weights=np.random.rand(3,1)

bias=np.random.rand(1)

lr=0.05



for epoch in range(20000):

    inputs=feature_set

    #print(inputs)



    #feedforward step1

    XW=np.dot(feature_set,weights)+ bias

    #print(XW)



    #feedforward step2

    Z=sigmoid(XW)



    #backprogpagation step1

    # error

    error = Z - labels



    #backprogpagation step2

    dcost_dpred=error

    dpred_dz=sigmoid_der(Z)



    z_delta=dcost_dpred*dpred_dz



    inputs=feature_set.T

    weights-=lr*np.dot(inputs,z_delta)



    for num in z_delta:

        bias=lr*num





#testing our neural network

single_point=np.array([0,1,0])

result=sigmoid(np.dot(single_point,weights)+bias)

print(result)