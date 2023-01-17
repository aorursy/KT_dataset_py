#importing numpy
import numpy as np
#sigmoid and darivative of sigmoid function
def sigdiv(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
# input arrays
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
# output array
Y = np.array([[0],
            [1],
            [1],
            [0]])
np.random.seed(0)
# Randomly initializing our weights with mean zero
W0 = 2*np.random.random((3,4)) - 1
W1 = 2*np.random.random((4,1)) - 1
for j in range(100000):
    #1 Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigdiv(np.dot(l0,W0))
    l2 = sigdiv(np.dot(l1,W1))
    
    #2 Calculating error
    l2_error = Y - l2
    
    #printing error
    if (j% 10000) == 0:
        print("Error iafter " + str(j) + " itration :" + str(np.mean(np.abs(l2_error))))
    
    # in what direction is the target value
    #3 calculating change to made in weights. 
    l2_delta = l2_error*sigdiv(l2,deriv=True)
    
    #4 how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(W1.T)
    
    # in what direction is the target value
    # calculating change to made in weights.
    l1_delta = l1_error * sigdiv(l1,deriv=True)
    
    #5 updating weights
    W1 += l1.T.dot(l2_delta)
    W0 += l0.T.dot(l1_delta)
#Output after training
l2