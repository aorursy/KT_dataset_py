import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Get Training & Test data

traindata = pd.read_csv('../input/train.csv')

imgTrainData = traindata.iloc[:,1:].values

imgTrainLabel = traindata.iloc[:,0].values

print(imgTrainData.shape)



#testdata = pd.read_csv('../input/test.csv').values



#Reshape of training data to 28x28 img format

imgTrainData = imgTrainData.reshape(imgTrainData.shape[0],28,28)



#Plot a Sample Img

plt.imshow(imgTrainData[1],cmap=plt.get_cmap('gray'))

plt.title(imgTrainLabel[1])
#Activation Functions

def sigmoid(x):

    return (1/(1+np.exp(-x)))



def dsigmoid(x):

    return (sigmoid(x)*(1-sigmoid(x)))



def tanh(x):

    return (np.tanh(x))



def dtanh(x):

    return (1-(np.square(np.tanh(x))))



def relu(x):

    return (np.maximum(x,0))



def drelu(x):

    return (1 * (x > 0))



def lkrelu(x):

    return (np.maximum(x,0.01*x))



def sofmax(x):

    z = np.exp(x)/np.exp(x).sum(axis=0)

    return (z)

 

#Plot activation functions.

fig, axes = plt.subplots(nrows=1, ncols=4)

x=np.linspace(-10,10,100)

axes[0].plot(x,sigmoid(x))

axes[1].plot(x,tanh(x))

axes[2].plot(x,relu(x))

axes[3].plot(x,lkrelu(x))

fig.tight_layout()
imgTrainData = imgTrainData.reshape(imgTrainData.shape[0],28*28)



#Building Neural Network

#randomly initialize our weights.

w0 = 2*np.random.random((784,32)) - 1 

w1 = 2*np.random.random((32,16)) - 1 

w3 = 2*np.random.random((16,10)) - 1 



#Forward Propagation

l0 = imgTrainData

l1 = relu(np.dot(l0,w0))

l2 = sigmoid(np.dot(l1,w1))



#Softmax Regression.

lout = sofmax(np.dot(l2,w3)).argmax(axis=1)

print(lout)

   

    
