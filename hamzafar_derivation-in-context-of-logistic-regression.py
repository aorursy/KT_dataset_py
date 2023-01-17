# This is Python 3 environment 
# Call packages for onward use

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting graph
def sigmoid(z):
    # Takes input as z and return sogmoid of value
    s = 1 / (1 + np.exp(-z))
    return s
# gnerate data with two feature set and get label of xor function
x1, x2 = 1, 1
y = int(np.logical_xor(x1,x2))
print('actual value(y): ', y)

# define paramters i.e. weights and bias to random
w1, w2, b = 0.1, 0.5, 0.005

# print('Parameters Before update')
# print('w1: ', w1, 'w2: ', w2, 'b: ', b)

z = w1*x1 + w2*x2 + b

#activation of values 
a = sigmoid(z)
print('activation value: ', a)

# compute the loss of the function(since we have training exmple equal to 1, so cost==loss )
cost = -1 * (y * np.log(a) + (1 - y) * (np.log(1 - a)))  # compute cost
print('loss of function: ', cost)

# Store cost, activation, weights and bias to dictionary
my_dic = {}
my_dic['w1'] = w1
my_dic['w2'] = w2
my_dic['b'] = b
my_dic['activation'] = a
my_dic['cost'] = cost

# BACKWARD PROPAGATION (TO FIND GRAD)
dw1 = x1*(a-y)
dw2 = x2*(a-y)
db = a - y 
from IPython.display import HTML

# Youtube
HTML('<iframe width="800" height="400" src="https://www.youtube.com/embed/8mS1DlibKbI" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
# update parameter w1, w2 and b by the equation

lr =0.07 # learning rate

w1 = w1 - (lr*dw1)
w2 = w2 - (lr*dw2)
b = b - (lr*db)

z = w1*x1 + w2*x2 + b

#activation of values 
a = sigmoid(z)
print('activation value before update: ', my_dic['activation'])
print('activation value after update: ', a)
print('')

# compute the loss of the function(since we have training exmple equal to 1, so cost==loss )
cost = -1 * (y * np.log(a) + (1 - y) * (np.log(1 - a)))  # compute cost
print('loss of function before update: ', my_dic['cost'])
print('loss of function after update: ', cost)
def get_activation_loss(x1, x2, w1, w2, b):
    # this function compute activations, cost and z
        # x : input features
        # w : weight
        # b : bias
    z = w1*x1 + w2*x2 + b

    #activation of values 
    a = sigmoid(z)

    # compute the loss of the function(since we have training exmple equal to 1, so cost==loss )
    cost = -1 * (y * np.log(a) + (1 - y) * (np.log(1 - a)))  # compute cost
    
    return(a,cost, z)
def update_paramters(x1, x2, w1, w2, b, a, y, lr):
    # This function computes gradient of parmaters and then update them
    # returns upadated parameters weights and bias
        # x: input features
        # w: weights
        # b: bias
        # a: activation
        # y: actual label
        # lr: learning rate
      
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw1 = x1*(a-y)
    dw2 = x2*(a-y)
    db = a - y 

    # update parameter w1, w2 and b by the equation

    w1 = w1 - (lr*dw1)
    w2 = w2 - (lr*dw2)
    b = b - (lr*db)
    
    return(w1, w2, b)
def plt_res(lst, ylab, lr):
    #This will plot the list of values at y axis while x axis will contain number of iteration
    #lst: lst of action/cost
    #ylab: y-axis label
    #lr: learning rate
    plt.plot(lst)
    plt.ylabel(ylab)
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    plt.show()
# define paramters i.e. weights and bias to random to new random values
w1, w2, b = 0.04, 0.75, 0.0015
lst_cost = []
lst_activation = []
#In code below, we will update paramets about 3500 times.
num_iter = 3500
lr = 0.007

# gnerate data with two feature set and get label of xor function
x1, x2 = 1, 0
y = int(np.logical_xor(0,1))

print ('x1: ', x1, '; x2: ',x2)
print('xor value(y): ', y)

for i in range(num_iter):
    a,cost,z = get_activation_loss(x1, x2, w1, w2, b)
#     print('cost at iteration', i,': ', cost)
#     print('activation at iteration', i,': ', a)
    w1, w2, b = update_paramters(x1, x2, w1, w2, b, a, y, lr)
    lst_cost.append(cost)
    lst_activation.append(a)

plt_res(lst_cost, 'loss', lr)
plt_res(lst_activation,'activation', lr)
    