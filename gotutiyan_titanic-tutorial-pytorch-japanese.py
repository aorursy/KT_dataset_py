# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision

from torch.utils.data import DataLoader

from torchvision import datasets

import torchvision.transforms as transforms

import os

import time

import sys

import torch.quantization
train_df = pd.read_csv('../input/titanic/train.csv')

train_df.head()
test_df = pd.read_csv('../input/titanic/test.csv')

test_df.head()
train_df.info()
print(train_df.dtypes)
def process_df(df):

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

    df["Age"] = df["Age"].fillna(df["Age"].mean())

    df = df.replace("male", 0)

    df = df.replace("female", 1)

    return df



train_df = process_df(train_df)

test_df = process_df(test_df)

train_df.head()
class Dataset:

    def __init__(self, df):

        self.df = df

        self.X = self.df.drop(["Survived"], axis=1)

        self.Y = self.df["Survived"]

    

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, idx):

#         print(type(self.X.iloc[idx,:]))

#         print(type(self.Y.iloc[idx]))

        return self.X.iloc[idx,:].values, self.Y.iloc[idx]



train_dataset = Dataset(train_df)

# test_dataset = Dataset(test_df)

len(train_dataset)
BATCH_SIZE = 32

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



# xに入力，yにラベルが入る

for x,y in train_dataloader:

    print(x,y)

    break
class Net(nn.Module):

    def __init__(self, input_sz, hidden_sz, out_sz):

        super(Net, self).__init__()

        self.f1 = nn.Linear(input_sz, hidden_sz)

        self.f2 = nn.Linear(hidden_sz, out_sz)

        

    def forward(self, x):

        h1 = F.sigmoid(self.f1(x))

        y = self.f2(h1)

        

        return y



input_sz = 6

hidden_sz = 3

out_sz = 2

net = Net(input_sz, hidden_sz, out_sz)
learning_rate = 0.01

loss_func = nn.MSELoss(reduction="sum")

optimizer = optim.Adam(net.parameters(), lr=learning_rate)

epoch = 32

def train():

    for e in range(epoch):

        for X, labels in train_dataloader:

            T = convert_label_to_onehot(labels)

            y = F.softmax(net(X.float()), dim=1)

            loss = loss_func(y, torch.FloatTensor(T))

            loss.backward()

            optimizer.step()

            

def convert_label_to_onehot(labels):

    onehot = np.zeros((len(labels), max(labels)+1))

    idx = [(i, t.item()) for i, t in enumerate(labels)]

    for i in idx:

        onehot[i] = 1

    return onehot



train()
# torch.max()の簡単な説明

prob = torch.tensor([[0.1, 0.9],

                    [0.2, 0.8],

                    [0.6, 0.4]])

max, argmax = torch.max(prob, dim=1)

print("max\t",max)

print("argmax\t",argmax)              
def test():

    test_X = torch.tensor(test_df.iloc[:,:].values)

    test_Y = net(test_X.float())

    survived = torch.max(test_Y, dim=1)[1]

    test_paID = pd.read_csv('../input/titanic/gender_submission.csv')['PassengerId']

    sub_df = pd.DataFrame({"PassengerId":test_paID.values, "Survived":survived})

    print(sub_df)

    return sub_df

    

sub_df = test()

sub_df.to_csv("./submission.csv", index=False)

    