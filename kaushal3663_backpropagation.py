# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd
Boston = pd.read_csv('/kaggle/input/Boston.csv')
Boston
X = Boston.drop('medv',axis =1)

y= Boston['medv']
X.head()
y.head()
class NeuralNetwork(object):

    def __init__(self):

        #parameters

        self.inputSize = 14

        self.outputSize = 1

        self.hiddenSize = 15

        

        #weights

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer

        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

        

    def feedForward(self, X):

        #forward propogation through the network

        self.z = np.dot(X, self.W1) #dot product of X (input) and first set of weights (3x2)

        self.z2 = self.sigmoid(self.z) #activation function

        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of weights (3x1)

        output = self.sigmoid(self.z3)

        return output

        

    def sigmoid(self, s, deriv=False):

        if (deriv == True):

            return s * (1 - s)

        return 1/(1 + np.exp(-s))

    

    def backward(self, X, y, output):

        #backward propogate through the network

        self.output_error = y - output # error in output

        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        

        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error

        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error

        

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights

        self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights

        

    def train(self, X, y):

        output = self.feedForward(X)

        self.backward(X, y, output)
NN = NeuralNetwork()



for i in range(1000): #trains the NN 1000 times

    if (i % 100 == 0):

        print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))

NN.train(X, y)

        

print("Input: " + str(X))

print("Actual Output: " + str(y))
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("Predicted Output: " + str(NN.feedForward(X)))