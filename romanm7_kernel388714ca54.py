# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from torch import nn

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

def grad_example(x,w): #variant 3 

    w.requires_grad_(True)

    ex = torch.exp(-x.matmul(w))

    res = 1/(1+ex)

    res.backward()

    print(w.grad)

x = torch.tensor([1.,2.,3.])

w = torch.tensor([2.,3.,4.])

grad_example(x,w)
cover_df = pd.read_csv("../input/covertype_forest.csv")
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(cover_df["class"])

cover_target = label_encoder.transform(cover_df["class"])

print(cover_target)
from sklearn.model_selection import train_test_split

cover_df = cover_df.drop("class",axis=1)

df_train, df_test, y_train, y_test = train_test_split(cover_df, cover_target, test_size=0.15, stratify=cover_target)
to_normalize = [(i, col) for i, col in enumerate(cover_df.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия



print(columns_to_normalize)

print(idx_to_normalize)

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
class my_net:

    def __init__(self):

        self.w_inp = torch.randn(54,200).type(torch.FloatTensor).requires_grad_(True)

        self.b_inp = torch.randn(200).type(torch.FloatTensor).requires_grad_(True)

        self.w_out = torch.randn(200,7).type(torch.FloatTensor).requires_grad_(True)

        self.b_out = torch.randn(7).type(torch.FloatTensor).requires_grad_(True)

    def train(self, loader,lr,epoch):

        for i in range(epoch):

            for xx,yy in loader:

                if self.w_inp.grad is not None:

                    self.w_inp.grad.zero_()

                    self.b_inp.grad.zero_()

                    self.w_out.grad.zero_()

                    self.b_out.grad.zero_()

                h = xx.matmul(self.w_inp) + self.b_inp

                h = h.relu()

                out = h.matmul(self.w_out) + self.b_out

                out = out.log_softmax(dim = 1)

                loss = -(1/len(xx))*out[torch.arange(len(xx)),yy].sum()

                loss.backward()

                with torch.no_grad():

                    self.w_inp -= self.w_inp.grad*lr

                    self.b_inp -= self.b_inp.grad*lr

                    self.w_out -= self.w_out.grad*lr

                    self.b_out -= self.b_out.grad*lr

            print(i,"loss: ", loss)

    def pred(self,loader):

        res = []

        for xx,yy in loader:

                h = xx.matmul(self.w_inp) + self.b_inp

                h = h.relu()

                out = h.matmul(self.w_out) + self.b_out

                out = out.log_softmax(dim = 1)

                res.append(out.argmax(dim=1).numpy())

        return res

    def predict(self,xx):

        h = xx.matmul(self.w_inp) + self.b_inp

        h = h.relu()

        out = h.matmul(self.w_out) + self.b_out

        out = out.log_softmax(dim = 1)

        return out
net = my_net()
net.train(train_loader,0.5,20)
from sklearn.metrics import classification_report

y_true = []

y_pred = []

for xx,yy in test_loader:

    out = net.predict(xx)

    for i in out:

        y_pred.append(int(i.argmax()))

    for i in yy:

        y_true.append(int(i))

print(classification_report(y_pred,y_true))