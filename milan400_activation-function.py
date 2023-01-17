import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
def sigmoid(x):

    s = 1/(1+np.exp(-x))

    ds = s*(1-s)

    return(s,ds)

x = np.arange(-6,6,0.01)
sigmoid(x)[1].shape
#Setup centeres axes

fig, ax = plt.subplots(figsize=(9,5))



ax.spines['left'].set_position('center')

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')



ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')



#Create and show plot

ax.plot(x,sigmoid(x)[0], color="#307EC7", linewidth=3, label="sigmoid")

ax.plot(x,sigmoid(x)[1], color="#9621E2", linewidth=3, label="derivative")



ax.legend(loc="upper left", frameon=False)

fig.show()
def tanh_activation(z):

    s = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

    ds = 1-s**2

    return(s,ds)
z = np.arange(-4,4,0.01)
#Setup centeres axes

fig, ax = plt.subplots(figsize=(9,5))



ax.spines['left'].set_position('center')

ax.spines['bottom'].set_position('center')



ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')



ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')



#Create and show plot

ax.plot(z,tanh_activation(z)[0], color="#307EC7", linewidth=3, label="Tanh")

ax.plot(z,tanh_activation(z)[1], color="#9621E2", linewidth=3, label="derivative")



ax.legend(loc="upper left", frameon=False)

fig.show()
def Relu_activation(r):

    return(np.maximum(r,0))
r = np.arange(-4,4,1)

Relu_activation(r)
#Setup centeres axes

fig, ax = plt.subplots(figsize=(9,5))



ax.spines['left'].set_position('center')

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')



ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')



#Create and show plot

ax.plot(r,Relu_activation(r), color="#307EC7", linewidth=3, label="Relu")



ax.legend(loc="upper left", frameon=False)

fig.show()
def leakRelu_activation_1(r):

    return(np.where(r>0,r,r*0.1))



def leakRelu_activation_2(r):

    return(np.where(r>0,r,r*0.2))
#Setup centeres axes

fig, ax = plt.subplots(figsize=(9,5))



ax.spines['left'].set_position('center')

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')



ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')



#Create and show plot

ax.plot(r,leakRelu_activation_1(r), color="#307EC7", linewidth=3, label="Relu_0.1")

ax.plot(r,leakRelu_activation_2(r), color="#9621E2", linewidth=3, label="Relu_0.2")



ax.legend(loc="upper left", frameon=False)

fig.show()
def soft_max(r):

    s = np.exp(r)/sum(np.exp(r))

    return(s)
rs = np.arange(0,8,0.1)

rs_soft = soft_max(rs)

prob_sum = np.sum(rs_soft)

print(prob_sum)
#Setup centeres axes

fig, ax = plt.subplots(figsize=(9,5))



ax.spines['left'].set_position('center')

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')



ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')



#Create and show plot

ax.plot(rs,soft_max(rs), color="#307EC7", linewidth=3, label="Softmax")



ax.legend(loc="upper left", frameon=False)

fig.show()