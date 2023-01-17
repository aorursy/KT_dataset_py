import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

import pandas as pd

import random



# Importing the dataset

df = pd.read_csv('../input/Iris.csv')



# Dropping index

df = df.drop(['Id'], axis=1)



# Encoding class labels

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['Species']))}

df['Species'] = df['Species'].map(class_mapping)



# Spliting dataset into train and test

df = df.as_matrix()

np.random.shuffle(df)



X = df[:,:-1]

y = df[:,-1]



train_len = int(0.8 * len(y))

test_len = int(len(y) - train_len)



X_train = X[:train_len]

y_train = y[:train_len]



X_test = X[-test_len:]

y_test = y[-test_len:]



# Defining the network

class DynamicNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):

        super(DynamicNet, self).__init__()

        self.input_linear = torch.nn.Linear(D_in, H)

        self.middle_linear = torch.nn.Linear(H, H)

        self.output_linear = torch.nn.Linear(H, D_out)



    def forward(self, x):

        h_relu = F.relu(self.input_linear(x))

        for _ in range(random.randint(0, 3)):

            h_relu = self.middle_linear(h_relu).clamp(min=0)

        y_pred = self.output_linear(h_relu)

        return y_pred



D_in, H, D_out = 4, 20, 3

lr = 0.01

model = DynamicNet(4, 20, 3)



if torch.cuda.is_available():

    print('CUDA Available')

    model.cuda()



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# Trainning the model

num_epochs = 2000

for epoch in range(num_epochs):

    # Forward pass: Compute predicted y by passing x to the model

    if torch.cuda.is_available():

        x = Variable(torch.Tensor(X_train).cuda())

        y = Variable(torch.Tensor(y_train).cuda())

    else:

        x = Variable(torch.Tensor(X_train).float())

        y = Variable(torch.Tensor(y_train).long())    



    y_pred = model(x)



    # Compute and print loss

    loss = criterion(y_pred, y)



    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch) % 100 == 0:

        print('Epoch [%d/%d] Loss: %.4f' %(epoch + 1, num_epochs, loss.data[0]))

        

# Getting the predictions and the accuracy score

if torch.cuda.is_available():

    x = Variable(torch.Tensor(X_test).cuda())

    y = torch.Tensor(y_test).long()

else:

    x = Variable(torch.Tensor(X_test).float())

    y = torch.Tensor(y_test).long()



out = model(x)

_, predicted = torch.max(out.data, 1)



print('Accuracy of the network %d %%' % (100 * torch.sum(y==predicted) / len(y_test)))