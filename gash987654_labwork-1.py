# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import torch

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def grad(x,y):

    x.requires_grad_(True)

    y.requires_grad_(True)

    xx = x.sum()/len(x)

    yy = y.sum()/len(y)

    print(xx)

    print(yy)

    xx = x-xx

    yy = y-yy

    f = torch.cos(xx.matmul(yy))

    f.backward()

    print("x", x.grad)

    print("y", y.grad)

x = torch.tensor([1.23,5.55,7.23])

y = torch.tensor([6.23,3.565,1.83])

grad(x,y)
from sklearn.datasets import fetch_openml

from sklearn.metrics import classification_report

covertype = fetch_openml(data_id=180)

cover_df = pd.DataFrame(data=covertype.data, columns=covertype.feature_names)

cover_df.sample(3)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(covertype.target)

cover_target = label_encoder.transform(covertype.target)

from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(cover_df, cover_target, test_size=0.15, stratify=cover_target)

to_normalize = [(i, col) for i, col in enumerate(cover_df.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия



print(columns_to_normalize)

print(idx_to_normalize)

cover_df[columns_to_normalize].sample(4)
from torch.utils.data import TensorDataset,DataLoader

tensor_train = torch.from_numpy(df_train.values).type(torch.FloatTensor)

tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)

train_mean = torch.mean(tensor_train[:,idx_to_normalize], dim=0)

train_std = torch.std(tensor_train[:,idx_to_normalize], dim=0)

tensor_train[:,idx_to_normalize] -= train_mean

tensor_train[:,idx_to_normalize] /= train_std

tensor_test[:,idx_to_normalize] -= train_mean

tensor_test[:,idx_to_normalize] /= train_std

y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)

y_test_tensor = torch.from_numpy(y_test).type(torch.LongTensor)

train_ds = TensorDataset(tensor_train, y_train_tensor)

test_ds = TensorDataset(tensor_test, y_test_tensor)

train_loader = DataLoader(train_ds,batch_size=256, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=256)

for xx, yy in train_loader:

    print(xx)

    print(yy)

    break
class nn:

    def __init__(self):

        self.aw  = torch.randn(54,1000).requires_grad_(True)

        self.ab      = torch.randn(1000).requires_grad_(True)

        self.bw  = torch.randn(1000,7).requires_grad_(True)

        self.bb      = torch.randn(7).requires_grad_(True)



    def train(self, epoch, train_loader,lr):

        loss_pred = 1000

        for i in range(epoch):

            for xx,yy in train_loader:

                if self.bb.grad is not None:

                    self.aw.grad.zero_()

                    self.ab.grad.zero_()

                    self.bw.grad.zero_()

                    self.bb.grad.zero_()

                hidden = xx.matmul(self.aw) + self.ab

                hidden = hidden.relu()

                out = hidden.matmul(self.bw) + self.bb

                out = out.log_softmax(dim = 1)

                loss = -(1/len(xx))*out[torch.arange(len(xx)),yy].sum()

                loss.backward()

                with torch.no_grad():

                    self.aw -= self.aw.grad

                    self.ab -= self.ab.grad

                    self.bw -= self.bw.grad

                    self.bb -= self.bb.grad

            print(i, loss.item())

        

    def pred(self,loader):

        predicted = []

        y_all = []

        for xx,yy in loader:

            hidden = xx.matmul(self.aw) + self.ab

            hidden = hidden.relu()

            out = hidden.matmul(self.bw) + self.bb

            out = out.log_softmax(dim = 1)

            out = out.argmax(dim=1)

            j = 0

            for k in yy:

                y_all.append(k)

                predicted.append(out[j])

                j+=1

        print(classification_report(predicted,y_all))

            
network = nn()

network.train(10,train_loader,0.001)
network.pred(test_loader)