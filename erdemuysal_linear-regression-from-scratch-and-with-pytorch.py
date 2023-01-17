# Uncomment the command below if Numpy or PyTorch is not installed

# !conda install numpy pytorch cpuonly -c pytorch -y
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torch.utils.data import DataLoader, TensorDataset, random_split
# Hyperparameters

split_rate = 0.2

batch_size = 64

learning_rate = 1e-4



# Other constants

DATA_FILENAME = 'housing.csv'

TARGET_COLUMN = 'median_house_value'  # Target to predict
df = pd.read_csv('../input/california-housing-prices/housing.csv')

df.head()  # Print first 5 row of the data frame, 5 is default value but desired value can be passed as an argument
df.columns, len(df.columns)
df.replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaNs

df.dropna(inplace=True)  # Drop NaN values from the dataframe
df.describe()
df.info()
# Inputs: (longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income)

# Convert from Pandas dataframe to numpy arrays

inputs = df.drop([TARGET_COLUMN, 'ocean_proximity'], axis=1).to_numpy()  # Drop TARGET_COLUMN since it is target, can not be used as input

targets = df[[TARGET_COLUMN]].to_numpy()                                 # Drop 'ocean_proximity' since it is not numerical value

inputs.shape, targets.shape
input_size = inputs.shape[1]  # 8

output_size = targets.shape[1]  # 1

input_size, output_size
inputs = torch.from_numpy(inputs)

inputs = inputs.float()

targets = torch.from_numpy(targets)

targets = targets.float()
inputs_mean = torch.mean(inputs, axis=0)

targets_mean = torch.mean(targets, axis=0)

inputs = inputs / inputs_mean  # Noramalize

#targets = targets / targets_mean  # Normalize
w = torch.randn(output_size, input_size, requires_grad=True)  # Weights

b = torch.randn(output_size, requires_grad=True)  # Biases

w, w.shape, b, b.shape
def model(x):

    return x @ w.t() + b
# Generate predictions

predictions = model(inputs)

predictions, targets # Compare with targets
# MSE loss

def mse(t1, t2):

    diff = t1 - t2

    return torch.sum(diff ** 2) / diff.numel()
# Compute loss

loss = mse(predictions, targets)

loss
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

predictions = model(inputs)

predictions
# Calculate the loss

loss = mse(predictions, targets)

loss
# Compute gradients

loss.backward()

w.grad, b.grad
# Adjust weights & reset gradients

with torch.no_grad():

    w -= w.grad * learning_rate

    b -= b.grad * learning_rate

    w.grad.zero_()

    b.grad.zero_()
w, b
# Calculate loss

predictions = model(inputs)

loss = mse(predictions, targets)

loss
# Train for 1000 epochs

for i in range(1000):

    preds = model(inputs)

    loss = mse(preds, targets)

    loss.backward()

    with torch.no_grad():

        w -= w.grad * learning_rate

        b -= b.grad * learning_rate

        w.grad.zero_()

        b.grad.zero_()
# Calculate loss

predictions = model(inputs)

loss = mse(predictions, targets)

loss
# Predictions

predictions
# Targets

targets
# Define dataset

dataset = TensorDataset(inputs, targets)

dataset[0:2]
# Define data loader

train_loader = DataLoader(dataset, batch_size, shuffle=True)
for xb, yb in train_loader:

    print(xb)

    print(yb)

    break
# Define model

model = nn.Linear(input_size, output_size)

print(model.weight)

print(model.bias)
# Parameters

list(model.parameters())
# Generate predictions

predictions = model(inputs)

predictions
# Define loss function

criterion = F.mse_loss
loss = criterion(model(inputs), targets)

loss
# Define optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Utility function to train the model

def fit(num_epochs, model, criterion, optimizer, train_loader):

    # Repeat for given number of epochs

    for epoch in range(num_epochs):

        # Train with batches of data

        for xb,yb in train_loader:

            # 1. Generate predictions

            pred = model(xb)

            # 2. Calculate loss

            loss = criterion(pred, yb)

            # 3. Compute gradients

            loss.backward()

            # 4. Update parameters using gradients

            optimizer.step()

            # 5. Reset the gradients to zero

            optimizer.zero_grad()

        # Print the progress

        if (epoch+1) % 10 == 0:

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
fit(100, model, criterion, optimizer, train_loader)
# Generate predictions

predictions = model(inputs)

predictions
# Compare with targets

targets