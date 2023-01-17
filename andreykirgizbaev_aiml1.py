import torch

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import torch

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def grad(x): # max(x_i)

    x.requires_grad_(True)

    f = torch.max(x)

    f.backward()

    print(x.grad)

x = torch.tensor([10.,2.,55.,1.,9])

grad(x)
cover_df = pd.read_csv("../input/covertype.csv")

targ = cover_df["class"]

cover_df = cover_df.drop("class",axis=1)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(targ)

cover_target = label_encoder.transform(targ)

from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(cover_df, cover_target, test_size=0.15, stratify=cover_target)

tensor_train = torch.from_numpy(df_train.values).type(torch.FloatTensor)

tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)

y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)

y_test_tensor = torch.from_numpy(y_test).type(torch.LongTensor)
to_normalize = [(i, col) for i, col in enumerate(cover_df.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия



print(columns_to_normalize)

print(idx_to_normalize)



train_mean = torch.mean(tensor_train[:,idx_to_normalize], dim=0)

train_std = torch.std(tensor_train[:,idx_to_normalize], dim=0)



tensor_train[:,idx_to_normalize] -= train_mean

tensor_train[:,idx_to_normalize] /= train_std

tensor_test[:,idx_to_normalize] -= train_mean

tensor_test[:,idx_to_normalize] /= train_std
from torch.utils.data import TensorDataset,DataLoader

train_ds = TensorDataset(tensor_train, y_train_tensor)

test_ds = TensorDataset(tensor_test, y_test_tensor)

train_loader = DataLoader(train_ds,batch_size=256, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=256)
for xx, yy in train_loader:

    print(xx.size())

    print(xx)

    print(yy)

    break
from sklearn.metrics import classification_report



class net:

    def __init__(self,x,hid,out):

        self.hid_layer  = torch.randn(x,hid)

        self.out_layer  = torch.randn(hid,out) 

        self.b_hid      = torch.randn(hid)

        self.b_out      = torch.randn(out)



    def train(self, epoch, train_loader):

        h = 0.5

        ep = epoch

        loss_pred = 1000

                

        self.hid_layer.requires_grad_(True)

        self.out_layer.requires_grad_(True)

        self.b_hid.requires_grad_(True)

        self.b_out.requires_grad_(True)

        

        for i in range(ep):

            for xx,yy in train_loader:

                if self.hid_layer.grad is not None:

                    self.hid_layer.grad.zero_()

                    self.out_layer.grad.zero_()

                    self.b_hid.grad.zero_()

                    self.b_out.grad.zero_()

                

                hid_out = xx.matmul(self.hid_layer) + self.b_hid

                hid_out = hid_out.relu()

                

                out = hid_out.matmul(self.out_layer) + self.b_out

                out = out.log_softmax(dim = 1)

                

                loss = -(1/len(xx))*out[torch.arange(len(xx)),yy].sum()

                loss.backward()

                

                with torch.no_grad():

                    self.hid_layer  -= h * self.hid_layer.grad

                    self.out_layer  -= h * self.out_layer.grad

                    self.b_hid      -= h * self.b_hid.grad

                    self.b_out      -= h * self.b_out.grad

            

            print(loss)

           

        

    def pred(self,x,y):       

        hid_out = x.matmul(self.hid_layer) + self.b_hid

        hid_out = hid_out.relu()

        out = hid_out.matmul(self.out_layer) + self.b_out

        out = out.log_softmax(dim = 1)

        

        pred = out.argmax(1)

        pred = pred.cpu()

        return pred
n = net(54, 540, 7)

n.train(25, train_loader)
prediction = n.pred(tensor_test, y_test_tensor)

print(classification_report(prediction,y_test_tensor))