# This Notebook computes cost function associated with a small set of data for linear regression

# You can Change value of Variable "X" and Variable "Y" in order to scale it according to your needs

# It takes Theta0 and Theta1 as inputs and plots your data along with linear model having variable Theta0 and Theta1

# it computes cost function of linear model having parameters Theta0 and Theta1. 

# try experimenting with notebook with different values of Theta0 , Theta1 , X & Y 



"""

Created on Sun Mar  8 14:44:48 2020



@author: shoaib.zafer

"""

import matplotlib.pyplot as plt

import numpy as np

# original data set

X = np.array([5, 3, 0,4])

print("X=",X)

Y = np.array([4, 4, 1,3])

print("Y=",Y)

theta0= 0

theta1= 1

theta0= float(theta0)

theta1=float(theta1)

h0=np.array(theta1*X) 

h0=h0+theta0

plt.scatter(X,Y)

plt.plot(X,h0,"g")

plt.show()

print("h0=",h0) 

inner_sum0=(h0-Y)**2

print( "m=",len (inner_sum0))

cost_function0=np.sum(inner_sum0)/(2*len(inner_sum0))

print("cost function=",cost_function0)