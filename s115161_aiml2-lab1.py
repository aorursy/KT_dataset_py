# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import torch

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def grad(w, x):

        w.requires_grad_(True)

        wx=torch.dot(w, x)

        ex=torch.exp(-wx)

        res=1/(1+ ex)

        res.backward()

        print(w.grad)
grad(torch.tensor([1.,2.,3.]), torch.tensor([1.,2.,3.]))
from sklearn.preprocessing import LabelEncoder

x_full = pd.read_csv("../input/dataset_184_covertype.csv")

y_full = x_full["class"]



label_encoder = LabelEncoder().fit(y_full)

x_full = x_full.drop("class",axis=1)

y_full = label_encoder.transform(y_full)

print(x_full.shape)

print(y_full.shape)
from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.15, stratify=y_full)



to_normalize = [(i, col) for i, col in enumerate(x_full.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия



x_full[columns_to_normalize].sample(4)
from torch.utils.data import TensorDataset,DataLoader

tensor_train = torch.from_numpy(df_train.values).type(torch.FloatTensor)

tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)

train_mean = torch.mean(tensor_train[:,idx_to_normalize], dim=0)

train_std = torch.std(tensor_train[:,idx_to_normalize], dim=0)

tensor_train[:,idx_to_normalize] -= train_mean

tensor_train[:,idx_to_normalize] /= train_std

tensor_test[:,idx_to_normalize] -= train_mean

tensor_test[:,idx_to_normalize] /= train_std

print(tensor_train[:3])
y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)

y_test_tensor = torch.from_numpy(y_test).type(torch.LongTensor)

train_ds = TensorDataset(tensor_train, y_train_tensor)

test_ds = TensorDataset(tensor_test, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=256)
for xx, yy in train_loader:

    print(xx[0])

    print(yy[0])

    break
from sklearn.metrics import classification_report

class network:

    def __init__(self,dense):

        self.hidw = torch.randn(dense[0], dense[1])

        self.hidb = torch.randn(dense[1])

        self.hidw.requires_grad_(True)

        self.hidb.requires_grad_(True)

        self.outw = torch.randn(dense[1], dense[2])

        self.outb = torch.randn(dense[2])

        self.outw.requires_grad_(True)

        self.outb.requires_grad_(True)

        

    def fit(self, train_loader,lr,epoch):

        self.lr = 0.01

        for i in range(epoch):

            for xx, yy in train_loader:

                hid = (self.hidb + xx.matmul(self.hidw)).relu()

                res = (self.outb +hid.matmul(self.outw)).log_softmax(dim = 1)

                criterion = -(1/len(xx))*res[torch.arange(len(xx)),yy].sum()

                criterion.backward()

                with torch.no_grad():

                    self.hidw  -= lr*self.hidw.grad

                    self.hidb  -= lr*self.hidb.grad

                    self.outw  -= lr*self.outw.grad

                    self.outb  -= lr*self.outb.grad

                self.hidw.grad.zero_()

                self.hidb.grad.zero_()

                self.outw.grad.zero_()

                self.outb.grad.zero_()

            print("Train loss=",criterion)

            

    def report(self,loader):

        def toarray(x,y):

            for i in x:

                y.append(i)

        true_labels = []

        pred_labels = []

        for xx, yy in train_loader:

            hid = (self.hidb + xx.matmul(self.hidw)).relu()

            res = (self.outb +hid.matmul(self.outw)).log_softmax(dim = 1)

            res = res.argmax(dim=1)

            toarray(res.numpy(),pred_labels)

            toarray(yy.numpy(),true_labels)

        return classification_report(pred_labels,true_labels)
dense = [54,256,7]

clf = network(dense)

clf.fit(train_loader,0.1,11)

print(clf.report(test_loader))