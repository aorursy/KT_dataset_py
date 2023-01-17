import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch
w_list = []

mse_list=[]
# Input data



x_data = [1.0, 2.0, 3.0]

y_data = [2.0, 4.0, 6.0]

w = 1.0
# Function for forward pass to predict y

def forward(x):

    return x*w
# Function to calcuate the loss of the model

# Loss is the square of difference of prediction and actual value



def loss(x,y):

    y_pred = forward(x)

    return (y_pred-y)**2
# Function to calcualte the gradient for w to be updated and get min loss.

# y_pred closer to y



# Gradient = derivative of the loss for constant x and y



# We are going to use a as 0.01 for starters



def gradient(x,y):

    return 2*x*(x*w-y)
# Training loop



print('Predict (before training)', 4, forward(4))



# Training loop



for epoch in range(100):

    l_sum=0

    for x_val, y_val in zip(x_data, y_data):

        grad = gradient(x_val, y_val)

        w = w-0.01*grad

        print('\tgrad: ', x_val, y_val, grad)

        l=loss(x_val, y_val)

        l_sum+=l

        

    print('Progress: ', epoch, 'w=', w, 'loss=', l)

    w_list.append(w)

    mse_list.append(l_sum/3)

    

    

print('Predict (After training)', '4 hours', forward(4))    
plt.plot(w_list, mse_list)

plt.ylabel('Loss')

plt.xlabel('w')

plt.show()