import sys

import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch

import torch.nn.functional as F

import torch.nn as nn

from torch import optim

from tqdm import tqdm



print('Python: {}'.format(sys.version))

print('Pandas: {}'.format(pd.__version__))

print('Numpy: {}'.format(np.__version__))

print('Sklearn: {}'.format(sklearn.__version__))

print('Torch: {}'.format(torch.__version__))
#Importing the Dataset

data_frame = pd.read_csv("../input/heart.csv")
# print the shape of the DataFrame

print("Shape of Data before Train_Test split: {}".format(data_frame.shape))

print(data_frame.head())
# plot histograms for each variable

data_frame.hist(figsize = (12, 12))

plt.show()
# create X and Y datasets for training

X = np.array(data_frame.drop(['target'], 1))

Y = np.array(data_frame['target'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
#Numpy to torch

X_train, X_test, Y_train, Y_test = map(torch.tensor, (X_train, X_test, Y_train, Y_test))

X_train = X_train.float()

Y_train = Y_train.long()

X_test = X_test.float()

Y_test = Y_test.long()

print("Train data: {}".format(X_train.shape,X_train.dtype, Y_train.shape,Y_train.dtype))

print("Test dara: {}".format(X_test.shape,X_test.dtype, Y_test.shape,Y_test.dtype))
#Class level Setup

class HeartDiseasePrediction(nn.Module):



    def __init__(self):

        super().__init__()

        torch.manual_seed(0)

        self.net = nn.Sequential(

            nn.Linear(13,16),

            nn.ReLU(),

            nn.Linear(16,8),

            nn.ReLU(),

            nn.Linear(8, 2),

            nn.Softmax()

        )



    def model(self, X):

        return self.net(X)



    def accuracy(self, y_hat, y):

        y_pred = torch.argmax(y_hat, dim=1)

        return (y == y_pred).float().mean()





    def fit(self, x, y, opt, loss_fn_CrossENtropy, epochs):

        acc_err = []

        loss_err = []

        for epochs in tqdm(range(epochs)):

            loss = loss_fn_CrossENtropy(obj.model(x), y)

            loss.backward()

            opt.step()

            opt.zero_grad()

            acc_err.append(obj.accuracy(obj.model(x), y))

            loss_err.append(loss.item())

        return loss_err, acc_err



    def predict(self, X_test, Y_test):

        a = obj.model(X_test)

        b = torch.argmax(a, dim=1)

        print("Accuracy of Test data:{}".format((Y_test == b).float().mean()))
#Creting object to call the functions

obj = HeartDiseasePrediction()
# Using Adam Gradient Descent to minismise the loss

opt = optim.Adam(obj.parameters(), lr=.001)

loss_err_train, acc_err_train = obj.fit(X_train, Y_train, opt, F.cross_entropy, 5000)
#Printing Loss before and after training

print("Loss before training ", loss_err_train[0])

print("Loss after training ", loss_err_train[-1])
#Calling Predict Function to measure Accuracy after model got trained

loss_err_test = obj.predict(X_test, Y_test)
#Visualisation of Accuracy and Loss

plt.subplot(1, 2, 1)

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('epoch')

plt.plot(loss_err_train, 'r-')

plt.legend(['train'])

plt.subplot(1, 2, 2)

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('epoch')

plt.plot(acc_err_train, 'g-')

plt.legend(['train'])

plt.show()