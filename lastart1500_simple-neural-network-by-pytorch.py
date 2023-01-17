import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
train = pd.read_csv('../input/train.csv')
head_name = train.columns.values.tolist()
categ_name = []
for i in head_name:
    if train[i].value_counts().shape[0] <= 2:
        categ_name.append(i)

train_category = train[categ_name]
train_category = train_category.dropna(how='any', axis=1)
x_train = train_category.iloc[:,:9]
y_train = train_category.iloc[:,9:10]
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


oe = OneHotEncoder()
x_train = oe.fit_transform(x_train).toarray()
y_train = np.array(y_train)
import torch.nn.functional as F #ネットワーク用の様々な関数

class NN(nn.Module):  
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

model = NN()
#y_train = oe.fit_transform(y_train).toarray()
train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             weight_decay=1e-5)

loss_list = []
epoch_nums = 2
for epoch in range(epoch_nums):
    for data in train_loader:
        x, y = data
        x =  Variable(x).float()
        y =  Variable(y).float()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data[0])
    
    print('epoch [{}/{}], loss: {:.6f}'.format(
        epoch + 1,
        epoch_nums,
        loss.data[0]))
    
torch.save(model.state_dict(), 'weight_AE.pth')
x_train_torch = torch.from_numpy(x_train)
x_train_torch = Variable(x_train_torch).float()
test = pd.read_csv('./test.csv')
head_name = test.columns.values.tolist()
categ_name = []
for i in head_name:
    if test[i].value_counts().shape[0] <= 2:
        categ_name.append(i)

test_category = test[categ_name]
test_category = test_category.dropna(how='any', axis=1)
x_test = test_category.iloc[:,:9]
oe = OneHotEncoder()
x_test = oe.fit_transform(x_test).toarray()
x_test = torch.from_numpy(x_test)
x_test = Variable(x_test).float()
prediction = model(x_test)
cpu_pred = prediction.cpu()
y_predict = cpu_pred.data.numpy()
y_predict.shape
sub_df = pd.DataFrame({"MachineIdentifier": test["MachineIdentifier"].values})
sub_df["HasDetections"] = y_predict
sub_df.to_csv("submit.csv", index=False)