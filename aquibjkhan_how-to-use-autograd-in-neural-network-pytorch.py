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
# require_grad = True : these are Weights and Biases are differentiable 

w1 = Variable(torch.rand(X_tr.shape[1], 10), requires_grad=True)

b1 = Variable(torch.ones(w1.shape[1]), requires_grad=True)

w2 = Variable(torch.rand(w1.shape[1], 3), requires_grad=True)

b2 = Variable(torch.ones(w2.shape[1]),  requires_grad=True)
lr = 0.001  # learning rate

for it in range(6000):

    # feed forward

    l0 = X_tr

    l1 = torch.sigmoid(torch.mm(l0, w1)+b1)              # activation layer = sigmoid(w.x+b)

    l2 = torch.sigmoid(torch.mm(l1, w2)+b2)

    out = l2



    # loss compute

    loss_mse = (out - y_tr).pow(2).sum()

    if it%1000 == 0:

        print(loss_mse)



    loss_mse.backward()

    

    with torch.no_grad():

        w1 -= lr*w1.grad

        b1 -= lr*b1.grad

        w2 -= lr*w2.grad

        b2 -= lr*b2.grad



        w1.grad.zero_()

        b1.grad.zero_()

        w2.grad.zero_()

        b2.grad.zero_()

# Let's do some prediction on our test data with our tiny Neural Network

prediction = torch.sigmoid(torch.mm(torch.sigmoid(torch.mm(X_te, w1)+b1), w2)+b2)
print(prediction)
correct_pred = 0

wrong_pred = 0

for it in zip(prediction, y_te) :

    if torch.argmax(it[0]) == torch.argmax(it[1]):

        correct_pred +=1

    else:

        wrong_pred +=1



print("Correct Prediction {}".format(correct_pred))

print("Wrong Prediction {}".format(wrong_pred))

    