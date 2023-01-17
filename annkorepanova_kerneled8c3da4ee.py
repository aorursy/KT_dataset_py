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
from sklearn.datasets import fetch_openml
covertype = fetch_openml(data_id=180)
cover_df = pd.DataFrame(data=covertype.data, columns=covertype.feature_names)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(covertype.target)
cover_target = label_encoder.transform(covertype.target)
from sklearn.model_selection import train_test_split
df_train, df_test, y_train, y_test = train_test_split(cover_df, cover_target, test_size=0.15, stratify=cover_target)
to_normalize = [(i, col) for i, col in enumerate(cover_df.columns)

                        if not col.startswith('wilderness_area') and not col.startswith('soil_type')]

idx_to_normalize = [i for i,col in to_normalize] #номера столбцов

columns_to_normalize = [col for i, col in to_normalize] #названия



#print(columns_to_normalize)

#print(idx_to_normalize)
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
lr = 0.7

batch_size = 256

lay_hide = batch_size*10

epoches = 10
train_loader = DataLoader(train_ds,batch_size, shuffle=True)

test_loader = DataLoader(test_ds, batch_size)
lay_first = len(tensor_train[0])

lay_out = len(label_encoder.classes_)
w1 = torch.rand((lay_first, lay_hide), requires_grad=True)

w2 = torch.rand((lay_hide,lay_out), requires_grad=True)

b1 = torch.rand(lay_hide, requires_grad=True)

b2 = torch.rand(lay_out, requires_grad=True)
wm = [w1,w2]

bm = [b1,b2]
Lmin = None 



for epoche in range(epoches):

    loss = 0

    for xx, yy in train_loader:

        if w1.grad is not None: 

            w1.grad.zero_()

            w2.grad.zero_() 

            b1.grad.zero_()

            b2.grad.zero_()

            

        h = xx.mm(w1) + b1

        h=torch.clamp(h, min=0, max=10000)

        #или nn.relu()

        

        p = (h.mm(w2) + b2).log_softmax(1)

        L = -torch.mean(p[torch.arange(p.shape[0]), yy])

        L.backward() 

        loss+=L

        

        #попытка сделать лог.софтмакс

        #sm = torch.zeros(P.shape[0],P.shape[1])

        #for i in range (P.shape[0]):       

        #    softmax_sum = torch.sum(torch.exp(P[i]))

        #    for j in range (P.shape[1]):

        #        ex = torch.exp(P[i][j])

        #        sm[i][j] = ex.item()/softmax_sum.item()

        #L = -torch.mean(torch.log(sm[torch.arange(sm.shape[0]), yy]))

        

        with torch.no_grad():

            w1 -= lr * w1.grad

            w2 -= lr * w2.grad

            b1 -= lr * b1.grad

            b2 -= lr * b2.grad

       

    loss /= yy.shape[0]

    print("epoch ",epoche+1," = ",loss)       

    if (Lmin!=None and Lmin>L):

        Lmin = L

        wm[0] = w1

        wm[1] = w2

        bm[0] = b1

        bm[1] = b2
from sklearn.metrics import classification_report

result = torch.tensor(y_test_tensor.shape)



for xx, yy in test_loader:  

    h = xx.mm(wm[0]) + bm[0]

    h = torch.clamp(h, min=0, max=10000)

    p = h.mm(wm[1]) + bm[1]

    p = p.log_softmax(1)   

    result = torch.cat((result,torch.argmax(p,dim=1)))

            

result=result[1:result.shape[0]]

print(classification_report(result.tolist(), y_test_tensor.tolist()))
