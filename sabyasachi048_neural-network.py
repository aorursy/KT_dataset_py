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
def sigmoid(x):

    return 1/(1+np.exp(-1*x))

def sigmoid_derivative(x):

    k=sigmoid(x)

    return k*(1-k)
class NeuralNetwork:

    def __init__(self, x, y):

        self.input      = x

        self.weights1   = np.random.rand(self.input.shape[1],4) 

        self.weights2   = np.random.rand(4,1)                 

        self.y          = y

        self.output     = np.zeros(self.y.shape)



    def feedforward(self):

        self.layer1 = sigmoid(np.dot(self.input, self.weights1))

        self.output = sigmoid(np.dot(self.layer1, self.weights2))



    def backprop(self):

        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))



        self.weights1 += d_weights1

        self.weights2 += d_weights2
nn = NeuralNetwork(np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]),np.array([0,1,1,0]).reshape(4,1))

it = 15000

while it>0:

    nn.feedforward()

    nn.backprop()

    print(nn.output)

    it-=1

print("Final")

print(nn.output)