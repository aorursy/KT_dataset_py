import numpy as np

def binary_crossentropy(y, yhat):

  #code is derived from the piecewise function

  if y == 0:

    return -np.log(1.0-yhat)



  if y == 1:

    return -np.log(yhat)



y = 0 

yhat = 0.05 



print(f'Loss: {binary_crossentropy(y, yhat)}')

# Loading in the data

import sklearn

from sklearn.datasets import load_breast_cancer 

# Visualization

import matplotlib as mpl   

import matplotlib.pyplot as plt

import pandas as pd



# Building the network 

import numpy as np



# Progress Bar

import tqdm as tqdm



import warnings

warnings.filterwarnings("ignore") #supresses warnings
full_df = pd.read_csv('https://raw.githubusercontent.com/karthikb19/data/master/breastcancer.csv') #preprocessed data

full_df.drop(['Unnamed: 0'], inplace=True, axis=1)

start_index = 150 #@param {type:"slider", min:0, max:564, step:1}

full_df[start_index:start_index+5]
X_train = full_df.drop('target', inplace=False, axis=1) #remove 'target' column from input features

y_train = full_df['target'] #stores target (1 or 0) in a separate array



#since we shuffled, the index numbers were messed up, this resets them

X_train = X_train.reset_index(drop=True) 

y_train = y_train.reset_index(drop=True)



#convert to numpy arrays with float values

X_train = np.array(X_train, dtype=float)

y_train = np.array(y_train, dtype=float)



#reshape y_train to make matrix multiplication possible

y_train = np.array(y_train).reshape(-1, 1)

class Perceptron:

    def __init__(self, x, y):



        self.input = np.array(x, dtype=float) 

        self.label = np.array(y, dtype=float)

        self.weights = np.random.rand(x.shape[1], y.shape[1]) #randomly initialize the weights

        self.z = self.input@self.weights #dot product of the vectors

        self.yhat = self.sigmoid(self.z) #apply activation function



    

    def sigmoid(self, x):

        return 1.0/(1.0+np.exp(-x))



    def sigmoid_deriv(self, x):

        s = sigmoid(x)

        return s(1-s)



    def forward_prop(self):

        self.yhat = self.sigmoid(self.input @ self.weights) #@ symbol represents matrix multiplication (also works for vectors)

        return self.yhat



    def back_prop(self):

        gradient = self.input.T @ (-2.0*(self.label - self.yhat)*self.sigmoid(self.yhat))  #self.input is the x value



        self.weights -= gradient #process of finding the minimum loss
simple_nn = Perceptron(X_train, y_train)

training_iterations = 1000



history = [] #we will store how the mean squared error changes after each iteration in this array



def mse(yhat, y):

    sum = 0.0

    for pred, label in zip(yhat, y):

        sum += (pred-label)**2

    return sum/len(yhat)



for i in range(training_iterations):

    simple_nn.forward_prop()

    simple_nn.back_prop()

    yhat = simple_nn.forward_prop()

    history.append(mse(yhat, simple_nn.label))



    

    

yhat = simple_nn.forward_prop()









print(f'Final Mean Squared Error: {mse(yhat, simple_nn.label)}')
plt.plot(history)

plt.ylabel('Mean Squared Error')

plt.xlabel('Training Iteration')