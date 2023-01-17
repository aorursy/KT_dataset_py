# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the 

# files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Code from https://iamtrask.github.io/2015/07/12/basic-python-network/



# More advanced example: https://www.kaggle.com/xenocide/tensorflow-neural-network-tutorial-with-iris-data

# My version: https://www.kaggle.com/ingeon/tensorflow-neural-network-tutorial-with-iris-data/editnb



# Tensorflow NN for housing prices: https://www.kaggle.com/alphasis/a-very-simple-neural-network

# Stanford course CNN: http://cs231n.github.io

# Predicting hand written numbers: https://www.kaggle.com/statinstilettos/neural-network-approach

# Another: https://www.kaggle.com/kakauandme/tensorflow-deep-nn

# 1 hour NN tutorial: https://www.kaggle.com/c/titanic/discussion/25345







import numpy as np



# sigmoid function

def nonlin(x,deriv=False):

    if(deriv==True):

        return x*(1-x)

    return 1/(1+np.exp(-x))

    

# input dataset

X = np.array([  [0,0,1],

                [0,1,1],

                [1,0,1],

                [1,1,1] ])

    

# output dataset            

y = np.array([[0,0,1,1]]).T



# seed random numbers to make calculation

# deterministic (just a good practice)

np.random.seed(1)



# initialize weights randomly with mean 0

syn0 = 2*np.random.random((3,1)) - 1



for iter in range(1,1000):



    # forward propagation

    l0 = X

    l1 = nonlin(np.dot(l0,syn0))



    # how much did we miss?

    l1_error = y - l1



    # multiply how much we missed by the 

    # slope of the sigmoid at the values in l1

    l1_delta = l1_error * nonlin(l1,True)



    # update weights

    syn0 += np.dot(l0.T,l1_delta)



print( "Output After Training:")

print( l1)