import numpy as np
import pandas as pd
import torch
data = pd.read_csv('../input/hdfc-bank-dataset/01-07-2019-TO-29-06-2020HDFCBANKEQN.csv').head()
data = data[['Close Price', 'Total Traded Quantity', 'Turnover','Deliverable Qty']]
X = data.drop(['Close Price'], axis=1)
y = data['Close Price']
X = X.astype('double') #.double()
y = y.astype('double') 
print(X.head())
print(y.head())
X = X.values
y = y.values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
inputs = torch.from_numpy(X)
targets = torch.from_numpy(y)
inputs = inputs.double()
print(inputs.dtype)
print(targets.dtype)

targets
# Weights and biases
w = torch.randn(1, 3, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(w)
print(b)
print(w)
print(b)

def model(x):
    return x @ w.t() + b
preds = model(inputs)
print(preds)

# Generate predictions
preds = model(inputs)
print(preds)
print(inputs)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(inputs)

scaler.transform([1,-1])
print(inputs)
import torch.nn as nn
from torch.utils.data import TensorDataset
# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]
from torch.utils.data import DataLoader
# Define data loader
batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
for xb, yb in train_dl:
    print(xb)
    print(yb)
    break
# Define model
model = nn.Linear(3, 1)
print(model.weight)
print(model.bias)
# Parameters
list(model.parameters())
# Generate predictions
preds = model(xb)
preds
