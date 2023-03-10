import numpy as np

import torch
# Input (temp, rainfall, humidity)

inputs = np.array([[73, 67, 43], 

                   [91, 88, 64], 

                   [87, 134, 58], 

                   [102, 43, 37], 

                   [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)

targets = np.array([[56, 70], 

                    [81, 101], 

                    [119, 133], 

                    [22, 37], 

                    [103, 119]], dtype='float32')
# Convert inputs and targets to tensors

inputs = torch.from_numpy(inputs)

targets = torch.from_numpy(targets)

print(inputs)

print(targets)
# Weights and biases

w = torch.randn(2, 3, requires_grad=True)

b = torch.randn(2, requires_grad=True)

print(w)

print(b)
def model(x):

    return x @ w.t() + b
# Generate predictions

preds = model(inputs)

print(preds)
# Compare with targets

print(targets)
# MSE loss

def mse(t1, t2):

    diff = t1 - t2

    return torch.sum(diff * diff) / diff.numel()
diff=preds-targets

print(diff)
# Compute loss

loss = mse(preds, targets)

print(loss)
# Compute gradients

loss.backward()
# Gradients for weights

print(w)

print(w.grad)
w.grad.zero_()

b.grad.zero_()

print(w.grad)

print(b.grad)
# Generate predictions

preds = model(inputs)

print(preds)
# Calculate the loss

loss = mse(preds, targets)

print(loss)
# Compute gradients

loss.backward()

print(w.grad)

print(b.grad)
# Adjust weights & reset gradients

with torch.no_grad():

    w -= w.grad * 1e-5

    b -= b.grad * 1e-5

    w.grad.zero_()

    b.grad.zero_()
print(w)

print(b)
# Calculate loss

preds = model(inputs)

loss = mse(preds, targets)

print(loss)
# Train for 100 epochs

for i in range(200):

    preds = model(inputs)

    loss = mse(preds, targets)

    loss.backward()

    with torch.no_grad():

        w -= w.grad * 1e-5

        b -= b.grad * 1e-5

        w.grad.zero_()

        b.grad.zero_()
# Calculate loss

preds = model(inputs)

loss = mse(preds, targets)

print(loss)
# Predictions

preds
# Targets

targets
!pip install jovian --upgrade -q
import jovian
jovian.commit()
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
from torch.utils.data import TensorDataset
# Define dataset

train_ds = TensorDataset(inputs, targets)

train_ds[0:3]
from torch.utils.data import DataLoader
# Define data loader

batch_size = 5

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
for xb, yb in train_dl:

    print(xb)

    print(yb)

    break
# Define model

model = nn.Linear(3, 2)

print(model.weight)

print(model.bias)
# Parameters

list(model.parameters())
# Generate predictions

preds = model(inputs)

preds
# Import nn.functional

import torch.nn.functional as F
# Define loss function

loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)

print(loss)
# Define optimizer

opt = torch.optim.SGD(model.parameters(), lr=1e-5)
# Utility function to train the model

def fit(num_epochs, model, loss_fn, opt):

    

    # Repeat for given number of epochs

    for epoch in range(num_epochs):

        

        # Train with batches of data

        for xb,yb in train_dl:

            

            # 1. Generate predictions

            pred = model(xb)

            

            # 2. Calculate loss

            loss = loss_fn(pred, yb)

            

            # 3. Compute gradients

            loss.backward()

            

            # 4. Update parameters using gradients

            opt.step()

            

            # 5. Reset the gradients to zero

            opt.zero_grad()

        

        # Print the progress

        if (epoch+1) % 10 == 0:

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
fit(100, model, loss_fn, opt)
# Generate predictions

preds = model(inputs)

preds
# Compare with targets

targets
import jovian
jovian.commit()