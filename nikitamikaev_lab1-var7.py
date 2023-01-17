# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset,DataLoader

from torch.nn import functional as F
def grad(x,y):

    x.requires_grad_(True)

    y.requires_grad_(True)

    f = torch.cos(x.matmul(y))

    f.backward()



x = torch.tensor([1.,2.,3.])

y = torch.tensor([4.,5.,6.])

grad(x,y)

print("X grad: ",x.grad.numpy())

print("Y grad: ",y.grad.numpy())
covertype_full_dataset = pd.read_csv("../input/covertype.csv")

y_all = covertype_full_dataset["class"]

x_all = covertype_full_dataset.drop("class",axis=1)

label_encoder = LabelEncoder().fit(y_all)

y_all = label_encoder.transform(y_all)



df_train, df_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, stratify=y_all)
tensor_train = torch.from_numpy(df_train.values).type(torch.FloatTensor)

tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)



idx_to_normalize = [i for (i, col) in enumerate(x_all)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]



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



train_loader = DataLoader(train_ds,batch_size=128, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=128)
from sklearn.metrics import classification_report,accuracy_score

class Perceptron:

    def __init__(self,seed):

        torch.manual_seed(seed)

        self.w1 = (torch.randn(54, 1000).requires_grad_(True))

        self.b1 = (torch.randn(1000).requires_grad_(True))

        self.w2 = (torch.randn(1000, 7).requires_grad_(True))

        self.b2 = (torch.randn(7).requires_grad_(True))

        

    def forward(self,x):

        h = (x.matmul(self.w1) + self.b1).relu()

        out = (h.matmul(self.w2)+ self.b2)

        return F.log_softmax(out)

    

    def train(self,train_loader,epoch,learning_rate):

        for i in range(epoch):

            all_preds = []

            correct_preds = []

            for xx, yy in train_loader:

                pred = self.forward(xx).requires_grad_(True)



                #with torch.enable_grad():

                loss = -(1/len(xx))*pred[torch.arange(len(xx)),yy].sum()

                if self.w1.grad is not None:

                    self.w1.grad.zero_()

                    self.b1.grad.zero_()

                    self.w2.grad.zero_()

                    self.b2.grad.zero_()

                loss.backward()

                

                all_preds.extend(pred.argmax(1).tolist())

                correct_preds.extend(yy.tolist())

                

                with torch.no_grad():

                    self.w1 -= learning_rate*self.w1.grad

                    self.b1 -= learning_rate*self.b1.grad

                    self.w2 -= learning_rate*self.w2.grad

                    self.b2 -= learning_rate*self.b2.grad

                

            

            ("Epoch: " + str(i) + ", loss=" + str(loss.item()) + ", accuracy=" + str(accuracy_score(all_preds,correct_preds)))

            print("Epoch: %d, loss=%f, test_accuracy=%f" % (i,loss.item(),accuracy_score(all_preds,correct_preds)))
nn = Perceptron(1609)

nn.train(train_loader,35,0.6)
all_preds = []

correct_preds = []

for xx,yy in test_loader:

    y_pred = nn.forward(xx)

    all_preds.extend(y_pred.argmax(1).tolist())

    correct_preds.extend(yy.tolist())

    

print(classification_report(all_preds,correct_preds))