import numpy as np 

import pandas as pd 

import sklearn.datasets

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import Dataset, DataLoader



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



from tqdm import tqdm



import os

print(os.listdir("../input"))

%matplotlib inline
X, y = sklearn.datasets.make_classification(n_samples=5000, n_features=20, n_informative=20, n_redundant=0, n_repeated=0, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)

X_test = torch.from_numpy(X_test).type(torch.FloatTensor)

y_train = torch.from_numpy(y_train).type(torch.FloatTensor)

y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
class CustomDataset(Dataset):

    def __init__(self, x_tensor, y_tensor):

        self.x = x_tensor

        self.y = y_tensor

    

    def __getitem__(self, index):

        return (self.x[index], self.y[index])

    

    def __len__(self):

        return len(self.x)
train_data = CustomDataset(X_train, y_train)

test_data = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
class NN_Classifier(nn.Module):

    

    def __init__(self):

        super(NN_Classifier, self).__init__()

        self.in_layer = nn.Linear(in_features=20, out_features=40)

        self.hidden_layer = nn.Linear(in_features=40, out_features=1)

        self.out_layer = nn.Sigmoid()

    

    def forward(self, x):

        x = self.in_layer(x)

        x = self.hidden_layer(x)

        x = self.out_layer(x)

        

        return x

    

    def predict(self, x):

        pred = self.forward(x)

        return pred
model = NN_Classifier()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.eval()
epochs = 1000
losses = []

for i in range(epochs):

    batch_loss = []

    for x_batch, y_batch in train_loader:

        y_pred = model.forward(x_batch)

        loss = criterion(y_pred, y_batch)

        losses.append(loss.item())

        batch_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if i%100==0:

        print('Epoch %s: ' % i + str(np.mean(batch_loss)))
y_hat = model.predict(X_test)
yhat = []

for i in y_hat:

    if i >= .5:

        yhat.append(1)

    else:

        yhat.append(0)
print(classification_report(y_test, yhat))