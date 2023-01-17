# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch



#x - number

#a, b - vector

#f(x, a, b) = max((a - x)**2 + b**2)



x = 5

#a, b = torch.randn(5), torch.randn(5)

a = torch.tensor([10., 2., 3., 5., 1.])

b = torch.tensor([1., 2., 3., 5., 1.])



def grad(x, a, b):

    a.requires_grad_(True)

    b.requires_grad_(True)

    f = torch.max(torch.pow((a-x),2) + torch.pow(b,2))

    

    f.backward()

    print("Grad A = {}".format(a.grad))

    print("Grad B = {}".format(b.grad))

  

grad(x, a, b)
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



covertype = pd.read_csv("../input/dataset_184_covertype.csv")



covertype_y = covertype["class"]

covertype_x = covertype.drop("class",axis=1)



label_encoder = LabelEncoder().fit(covertype_y)

cover_target = label_encoder.transform(covertype_y)



df_train, df_test, y_train, y_test = train_test_split(covertype_x, cover_target, test_size=0.15, stratify=cover_target)
from torch.utils.data import TensorDataset,DataLoader

tensor_train = torch.from_numpy(df_train.values).type(torch.FloatTensor)

tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)



to_normalize = [(i, col) for i, col in enumerate(covertype_x)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия





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

print(train_ds[400])
from torch.nn import functional as F

train_loader = DataLoader(train_ds,batch_size=32, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=32)

for xx, yy in train_loader:

    print(xx)

    print(yy)

    break
class NN:

    def __init__(self,input_size,hidden_size,output_size):

        self.hw = (torch.randn(input_size, hidden_size).requires_grad_(True))

        self.hb = (torch.randn(hidden_size).requires_grad_(True))

        

        self.outw = (torch.randn(hidden_size, output_size).requires_grad_(True))

        self.outb = (torch.randn(output_size).requires_grad_(True))

        

    def forward(self,x):

        #h = self.hw@x+self.hb

        h = x.matmul(self.hw)+self.hb

        h = h.relu()

        out = h.matmul(self.outw)+self.outb

        return F.log_softmax(out)

    

    def train(self,train_loader,epoch,lr):

        self.hw.to(torch.device('cuda'))

        self.hb.to(torch.device('cuda'))

        self.outw.to(torch.device('cuda'))

        self.outb.to(torch.device('cuda'))

        for i in range(epoch):

            for xx, yy in train_loader:

                #xx,yy = xx.cuda(),yy.cuda()

                xx.to(torch.device('cuda'))

                yy.to(torch.device('cuda'))

                



                

                pred = self.forward(xx)

                loss = -(1/len(xx))*pred[torch.arange(len(xx)),yy].sum()

                loss.backward()

                with torch.no_grad():

                    self.hw -= lr*self.hw.grad

                    self.hb -= lr*self.hb.grad

                    

                    self.outw -= lr*self.outw.grad

                    self.outb -= lr*self.outb.grad

                    

                self.hw.grad.zero_()

                self.hb.grad.zero_()

                

                self.outw.grad.zero_()

                self.outb.grad.zero_()

                

            print("Epoch: " + str(i) + ", loss=" + str(loss.item()))
model = NN(54,10000,7)

model.train(train_loader,15,1)

model.train(train_loader,10,0.1)

model.train(train_loader,10,0.01)

model.train(train_loader,10,0.001)
from sklearn.metrics import classification_report



def add(from_,to_):

    for i in from_:

        to_.append(i)



pred_y=[]

true_y=[]



for xx,yy in test_loader:

    y_pred = model.forward(xx).argmax(1)

    add(y_pred,pred_y)

    add(yy,true_y)

    



print(classification_report(pred_y,true_y))