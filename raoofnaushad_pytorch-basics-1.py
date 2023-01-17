!pip install torchvision
import torch
t1 = torch.tensor(4.)
t2 = torch.tensor([3., 2, 1, 8])
t3 = torch.tensor([[[1, 2, 3, 4],[2,3,4,5]],
                 [[1, 2, 3, 4],[2,3,4,5]],
                 [[1, 2, 3, 4],[2,3,4,5]]])
print(t1.dtype)
print(t2.dtype)
print(t3.dtype)

print(t1.shape)
print(t2.shape)
print(t3.shape)
### Creating tensors
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
## Operation
y = w * x + b
print(y)
## To compute the derivatives in case were we put required_grad = True can be called using .backward()
y.backward()
## Finding derivatives
print('dy/dx', x.grad)   ### No gradient will be there since we didn't mentioned requries_grad = True there
print('dy/dw', w.grad)
print('dy/db', b.grad)

import numpy as np

## Creating a numpy array
x = np.array([[1, 2], [2, 3]])
print(x)
## Converting numpy array to torch tensor
y = torch.from_numpy(x)
print(y)
print(x.dtype, y.dtype)
## Converting torch to numpy
z = y.numpy()
print(z)
## Intializing inputs and targets
inputs = np.array([[73, 67, 43],
                  [91, 88, 64],
                  [87, 134,  58],
                  [102, 43, 37],
                  [69, 96, 70]], dtype='float32')
targets = np.array([[56, 70],
                   [81, 101],
                   [119, 133],
                   [22, 37],
                   [103, 119]], dtype='float32')
## Getting torch tensors from numpy
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
############################################ Single Epoch ###########################################
## Intialising random weights and bias
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2,  requires_grad=True)
## Equating conditions
def model(x):
    y =  x @ w.t() + b
    return y
## MSE (Mean Square Error)
def mse(d1, d2):
    diff = d1 - d2
    avg_loss = torch.sum(diff*diff)/diff.numel()
    return avg_loss
## Predicting
predictions = model(inputs)
loss = mse(predictions, targets)
## For getting derivative
loss.backward()
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()
predictions = model(inputs)
loss = mse(predictions, targets)


########################  100 Epochs #####################
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()
#     print(loss)
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
print(preds)
print(targets)
    
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

## Convert numpy to torch tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
from torch.utils.data import TensorDataset, DataLoader

######################################## Single Epoch ###############################################

## Using TensorDataset allows rows from ip and targets to avail as tuples
train_dl = TensorDataset(inputs, targets)
## Dataloader load data from tensordataset according to a particular batch_size
data = DataLoader(train_dl, batch_size=5, shuffle=True)

## Linear Regression model in torch.nn
model = nn.Linear(3, 2)
# print(list(model.parameters()))
preds = model(inputs)
# print(preds)

## Loss Required from Functional library of torch
import torch.nn.functional as F
loss_func = F.mse_loss

print(loss_func(model(inputs), targets))
 

# importing optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

## Model 
def fit(num_epochs, model, data, opt, loss_fn):
    ## 100 epochs
    for epoch in range(num_epochs):
        ## Train with dataloader data
        for xb, yb in data:
            
            preds = model(xb)
            loss = loss_fn(preds, yb)
            # For taking derivative
            loss.backward()
            
            ## Optimizing parameters using gradients
            opt.step()
            
            ## Setting gradient to zero
            opt.zero_grad()
            
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
## Fitting the model
fit(100, model, data, opt, loss_func)

preds = model(inputs)
loss = loss_func(preds, targets)
print(loss)
print(preds)
print(targets)
