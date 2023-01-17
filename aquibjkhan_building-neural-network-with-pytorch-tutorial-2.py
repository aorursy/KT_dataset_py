# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import torch

from torch.autograd import Variable

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
# Loading IRIS data

data_iris = load_iris()
# Let's extract features and Targets variables

data = data_iris["data"]

target = data_iris["target"]



print("shape of the data: ", data.shape)

print("shape of the target: ", target.shape)
#Target are categorical variables let's create one-hot encoding

def one_hot_enc(cat_var):

    # Binary encoding

    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = cat_var.reshape(len(cat_var), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

labels = one_hot_enc(target)
# Let's convert our data into PyTorch tensors

X_train, X_test, y_train, y_test = train_test_split(list(data), list(labels), test_size=0.10, random_state=42)



X_tr = Variable(torch.tensor(X_train, dtype=torch.float))

X_te = Variable(torch.tensor(X_test, dtype=torch.float))

y_tr = Variable(torch.tensor(y_train, dtype=torch.float))

y_te = Variable(torch.tensor(y_test, dtype=torch.float))
# N: Batch size, D_in: Input dimension, H1: First hidden layer size, H2: Second hidden layer size, D_out: Output unit size

N, D_in, H1,H2, D_out = 8, 4, 10, 5, 3
# Creating sequential model

model = torch.nn.Sequential(

        torch.nn.Linear(D_in, H1),    

        torch.nn.Sigmoid(),           # First hidden layer

        torch.nn.Linear(H1, H2),      

        torch.nn.Sigmoid(),           # Second hidden layer

        torch.nn.Linear(H2, D_out),   

        torch.nn.Sigmoid()            # Third layer/ ouput 

)
model
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(6000):

    y_pred = model(X_tr)

    # computing the loss function

    loss = loss_fn(y_pred, y_tr)

    if t%1000 == 0:

        print(t, loss.item())

        

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
pred_from_model = model(X_te)
correct_pred = 0

wrong_pred = 0

for it in zip(pred_from_model, y_te) :

    if torch.argmax(it[0]) == torch.argmax(it[1]):

        correct_pred +=1

    else:

        wrong_pred +=1



print("Correct Prediction {}".format(correct_pred))

print("Wrong Prediction {}".format(wrong_pred))

    
D_in = 4

H1 = 10

H2 = 5

D_out = 3



class TwoLayerNet(torch.nn.Module):

    

    def __init__(self, D_in, H1,H2, D_out):

        

        super(TwoLayerNet, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H1)

        self.linear2 = torch.nn.Linear(H1, H2)

        self.linear3 = torch.nn.Linear(H2, D_out)

    

    def forward(self, x):

        h1_state = torch.sigmoid(self.linear1(x))

        h2_state = torch.sigmoid(self.linear2(h1_state))

        y_pred = torch.sigmoid(self.linear3(h2_state))        

        return y_pred    
# Building the model

model_NN = TwoLayerNet(D_in, H1, H2, D_out)

print(model_NN)
criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.SGD(model_NN.parameters(), lr=1e-3)

for t in range(6000):

    # Forward pass: Compute predicted y by passing x to the model

    y_pred = model_NN(X_tr)



    # Compute and print loss

    loss = criterion(y_pred, y_tr)

    if t % 1000 == 0:

        print("At iteration {} the loss is {}".format(t, loss.item()))



    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
pred_from_model = model_NN(X_te)
correct_pred = 0

wrong_pred = 0

for it in zip(pred_from_model, y_te) :

    if torch.argmax(it[0]) == torch.argmax(it[1]):

        correct_pred +=1

    else:

        wrong_pred +=1



print("Correct Prediction {}".format(correct_pred))

print("Wrong Prediction {}".format(wrong_pred))