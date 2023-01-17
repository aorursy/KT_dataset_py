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



def drvsigmoid(x):

    return (sigmoid(x)*(1-sigmoid(x)))





#Plot activation functions.

x=np.linspace(-10,10,100)

plt.plot(x,sigmoid(x))

plt.show()
