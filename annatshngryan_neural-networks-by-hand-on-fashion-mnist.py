import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch


import os
print(os.listdir("../input"))

X_train = pd.read_csv("../input/fashion-mnist_train.csv")
X_test = pd.read_csv("../input/fashion-mnist_test.csv")
X_train.head()
y_train = X_train["label"]
X_train= X_train.drop(columns=["label"])
X_train.shape
y_train.shape
y_test = X_test["label"]
X_test= X_test.drop(columns=["label"])
X_test.shape
y_test.shape
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
X_train = torch.tensor(X_train, dtype=torch.float).view(-1, 784)
X_test = torch.tensor(X_test, dtype=torch.float).view(-1, 784)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
X_train.shape
y_train.shape
#we're gonna have this much units in each layer: 784 -> 512 -> 256 -> 128 -> 10 
#10 is the number of possible labels that we have
#We initialize weights and b's, and write requires_grad = True, for it to authomatically 
#calculate the gradients. 
W1 = torch.randn(512, 784)/100       
W1.requires_grad = True
b1 = torch.zeros(512, requires_grad=True)

W2 = torch.randn(256, 512)/100
W2.requires_grad = True
b2 = torch.zeros(256, requires_grad=True)

W3 = torch.randn(128, 256)/100
W3.requires_grad = True
b3 = torch.zeros(128, requires_grad=True)

W4 = torch.randn(10, 128)/100
W4.requires_grad = True
b4 = torch.zeros(10, requires_grad=True)

params = [W1,b1, W2,b2, W3,b3, W4,b4]
#We go through the whole data "epoch" times. 
#It is 1 complete iteration.(One forward AND one backward pass)
#Batch size defines the number of observations which we will consider in one pass. 
epochs = 5
learning_rate = 0.01
batch_size = 64
m = 60_000

#Softmax assigns decimal probabilities to each class in a multi-class problem.
softmax = torch.nn.LogSoftmax()
loss_fun = torch.nn.NLLLoss()

#Z's are basically X*W+b for each layer. 
#tanh is the activation function here. 
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    for batch_i in range(0, m, batch_size):
        X_batch = X_train[batch_i:batch_i+batch_size]
        y_batch = y_train[batch_i:batch_i+batch_size]
        
        Z1 = torch.matmul(X_batch, W1.t()) + b1 
        A1 = torch.tanh(Z1)
        
        Z2 = torch.matmul(A1, W2.t()) + b2
        A2 = torch.tanh(Z2)
        
        Z3 = torch.matmul(A2, W3.t()) + b3
        A3 = torch.tanh(Z3) 
        
        Z4 = torch.matmul(A3, W4.t()) + b4 
        
        out = softmax(Z4)
        loss = loss_fun(out, y_batch)
        loss.backward()
        
        for p in params:
            p.data = (p.data - learning_rate*p.grad).data
            p.grad.zero_()
        
        _, predictions = torch.max(Z4, 1) 
        epoch_acc += torch.sum(predictions == y_batch).item()
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch}:", epoch_loss/(m//batch_size), epoch_acc/m)
    