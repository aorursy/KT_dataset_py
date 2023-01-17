#importing Numpy and PyTorch Libraries

import numpy as np

import torch
# Input(temp, rainfall, humidity)

inputs = np.array([[73, 67, 43], 

                   [91, 88, 64], 

                   [87, 134, 58], 

                   [102, 43, 37], 

                   [69, 96, 70]], dtype='float32')



# Output(apples, orange)

targets = np.array([[56, 70], 

                    [81, 101], 

                    [119, 133], 

                    [22, 37], 

                    [103, 119]], dtype='float32')





# inputs and targets into tensors

inputs = torch.from_numpy(inputs)

targets = torch.from_numpy(targets)

print("Input in tensor form : \n",inputs ,"\nTarget in tensor form: \n",targets)
# weights and biases

w = torch.randn(2,3, requires_grad = True)

b = torch.randn(2, requires_grad = True)

print(w,b)
# model function

def model(x):

    return x @ w.t() +b  # @ represents matrix multiplication in Pytorch.



# prediction 

preds = model(inputs)

preds
#compare preds with the targets

print(targets)
# Meas Squarred Error (Loss Function)

def mse(t1,t2):

    diff = (t1-t2)

    return torch.sum(diff**2)/diff.numel()



#Calculate loss b/wprediction and targets

loss = mse(preds,targets)

loss
#Compute gradients

loss.backward()



# Gradiants for weoghts

print(w.grad)

print(b.grad)
# weights and biases

print(w)

print(b)
# adjust weights and reset gradiants

with torch.no_grad():

    w -= w.grad * 1e-5

    b -= b.grad * 1e-5

    w.grad.zero_()

    b.grad.zero_()

    

    

# new weights and biases

print(w)

print(b)

    
# calculate loss

preds= model(inputs)

loss = mse(preds,targets)

loss
# Training for some number of epochs to reduce the loss.

for i in range(3000):

    preds = model(inputs)

    loss = mse(preds,targets)

    loss.backward()

    with torch.no_grad():

        w -= w.grad * 1e-5

        b -= b.grad * 1e-5

        w.grad.zero_()

        b.grad.zero_()

        

        

#Calculate loss after the running model for some epochs

preds = model(inputs)

loss = mse(preds, targets)

print(loss)
# compare the predicted values with the actual values.

print(preds)

print(targets)
#importing the torch.nn package from PyTorch, which contains utility classes for building neural networks

import torch.nn as nn



# Input (temp, rainfall, humidity)

inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 

                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 

                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 

                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 

                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 

                  dtype='float32')



# Targets (apples, oranges)

targets = np.array([[56, 70], [81, 101], [119, 133], 

                    [22, 37], [103, 119], [56, 70], 

                    [81, 101], [119, 133], [22, 37], 

                    [103, 119], [56, 70], [81, 101], 

                    [119, 133], [22, 37], [103, 119]], 

                   dtype='float32')



inputs = torch.from_numpy(inputs)

targets = torch.from_numpy(targets)
from torch.utils.data import TensorDataset, DataLoader





# Dataset 

train_ds = TensorDataset(inputs, targets)

train_ds[0:3]
#DataLoader

batchsize = 5

train_dl = DataLoader(train_ds, batchsize ,shuffle= True)

train_dl
#Look into one of the batch

for xb, yb in train_dl:

    print(xb)

    print(yb)

    break
# define Model

model = nn.Linear(3,2)   # 3 input variables amd 2 output variables.

print("Weights: \n",model.weight)

print("Biases: \n",model.bias)
# Parameters

list(model.parameters())
preds = model(inputs)

preds
import torch.nn.functional  as F



# loss function

loss_fn= F.mse_loss



#compute loss

loss = loss_fn(preds,targets)

loss
#define optimizer

opt= torch.optim.SGD(model.parameters(), lr = 1e-5)
def fit(epoches, model, loss_fn, opt, train_dl):

    for epoch in range(epoches):

        for xb,yb in train_dl:

            pred=model(xb)

            

            # compute loss

            loss=loss_fn(pred,yb)

            

            # compute gradients

            loss.backward()

            

            # update parameter using gradients

            opt.step()

            

            #reset gradient to zero

            opt.zero_grad()

            

        if (epoch+1) %10 ==0:

            print(f"Epoch[{epoch+1}/{epoches}], Loss: {loss.item()} ")
fit(500, model, loss_fn, opt,train_dl)
preds = model(inputs)

preds[0:5]
targets[0:5]