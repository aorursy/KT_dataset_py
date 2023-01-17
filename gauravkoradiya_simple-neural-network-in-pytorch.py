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
import torch
def activation(x):
    return 1/(1+torch.exp(-x))
### Generate some data
torch.manual_seed(7)
# Features are 5 random normal variables
features = torch.randn((1, 5))
print(features)
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
print(weights)
bias = torch.randn((1, 1))
print(bias)

weights=weights.view(5,1)
print(weights)
torch.mm(features,weights)
y = activation(torch.sum(features * weights) + bias)
y = activation((features * weights).sum() + bias)
y = activation(torch.mm(features, weights.view(5,1)) + bias)
### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
print(output)

