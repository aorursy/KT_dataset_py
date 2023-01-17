import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def grad_example(x):

    x.requires_grad_(True)

    res = torch.max(x)

    res.backward()

    print(x.grad)

x = torch.tensor([9.,8.,7.,5])

grad_example(x)

x = torch.tensor([9.,80.,7.,5])

grad_example(x)
from sklearn.datasets import fetch_openml

covertype = fetch_openml(data_id=180)

cover_df = pd.DataFrame(data=covertype.data, columns=covertype.feature_names)

cover_df.sample(10)
from sklearn.preprocessing import LabelEncoder

print(covertype.target)

label_encoder = LabelEncoder().fit(covertype.target)

cover_target = label_encoder.transform(covertype.target)

print(cover_df.shape)

from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(cover_df, cover_target, test_size=0.15, stratify=cover_target)

to_normalize = [(i, col) for i, col in enumerate(cover_df.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия
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
from torch.utils.data import TensorDataset,DataLoader

train_ds = TensorDataset(tensor_train, y_train_tensor)

test_ds = TensorDataset(tensor_test, y_test_tensor)

train_loader = DataLoader(train_ds,batch_size=64, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=64)
from sklearn.metrics import classification_report



class Net():

    def __init__(self, t1, t2, t3):

        self.w = [torch.randn((t1, t2)), torch.randn((t2,t3))]

        self.b = [torch.randn(t2), torch.randn(t3)]

        for i in range(len(self.w)):

            self.w[i].requires_grad_(True)

            self.b[i].requires_grad_(True)      

        

    def fit(self, train_loader, epoches, lr):  

        for epoche in range(epoches):

            batch_loss = 0

            for xx, yy in train_loader:

                if self.w[0].grad is not None:

                    for i in range(len(self.w)):

                        self.w[i].grad.zero_()

                        self.b[i].grad.zero_()

                outh = (xx.mm(self.w[0]) + self.b[0]).relu()

                outy = (outh.mm(self.w[1]) + self.b[1]).log_softmax(1)

                loss = -outy[torch.arange(len(xx)),yy].sum()/len(xx)

                loss.backward()  

                with torch.no_grad():

                    batch_loss += loss.item()

                    for i in range(len(self.w)):

                        self.w[i] -= lr * self.w[i].grad

                        self.b[i] -= lr * self.b[i].grad

                batch_loss /= len(xx)   

            print("Epoch {}, loss={}".format(epoche+1, batch_loss))

        

    def predict(self, test_loader):

        res = None

        with torch.no_grad():

            for xx, yy in test_loader:

                outh = (xx.mm(self.w[0]) + self.b[0]).relu()

                outy = (outh.mm(self.w[1]) + self.b[1]).log_softmax(1)

                if res is None:

                    res = outy.argmax(1)

                else:

                    res = torch.cat((res, outy.argmax(1)))

        print(classification_report(res.tolist(), y_test_tensor.tolist()))

        return res
n = Net(54, 100, 7)

n.fit(train_loader, 10, 0.06)
n.predict(test_loader)