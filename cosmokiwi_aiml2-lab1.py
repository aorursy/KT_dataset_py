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
print(type(covertype), covertype)
cover_df = pd.DataFrame(data=covertype.data, columns=covertype.feature_names)
cover_df.sample(10)
print(covertype.target)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(covertype.target)
print(label_encoder.classes_) 
cover_target = label_encoder.transform(covertype.target)
print(cover_target)
print(cover_df.shape)
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
print(tensor_train[:3])
tensor_test = torch.from_numpy(df_test.values).type(torch.FloatTensor)
train_mean = torch.mean(tensor_train[:,idx_to_normalize], dim=0)

train_std = torch.std(tensor_train[:,idx_to_normalize], dim=0)
print(train_mean, train_std)
tensor_train[:,idx_to_normalize] -= train_mean

tensor_train[:,idx_to_normalize] /= train_std

tensor_test[:,idx_to_normalize] -= train_mean

tensor_test[:,idx_to_normalize] /= train_std
print(tensor_train[:3])
y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)

y_test_tensor = torch.from_numpy(y_test).type(torch.LongTensor)
train_ds = TensorDataset(tensor_train, y_train_tensor)

test_ds = TensorDataset(tensor_test, y_test_tensor)
print(train_ds[400])
train_loader = DataLoader(train_ds,batch_size=256, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=256)
for xx, yy in train_loader:

    print(xx)

    print(yy)

    break
N = 25 #число нейронов скрытого слоя

wm = torch.randn(54, N) #m - скрытый слой

wy = torch.randn(N, 7)  #y -  выходной слой

bm = torch.randn(N)

by = torch.randn(7)

lr = 0.01



wm.requires_grad_(True)

wy.requires_grad_(True)

bm.requires_grad_(True)

by.requires_grad_(True)
for xx, yy in train_loader:

    with torch.no_grad():

        if wm.grad is not None:

            wm.grad = None

            wy.grad = None

            bm.grad = None

            by.grad = None

    

    fm = xx.mm(wm) + bm 

    fm = fm.relu()

    

    fy = fm.mm(wy) + by

    fy = fy.log_softmax(1)

    loss = -fy[torch.arange(len(xx)),yy].sum()/len(xx)

    loss.backward()

    print(loss)

        

    with torch.no_grad():

        wm -= wm.grad*lr

        wy -= wy.grad*lr

        bm -= bm.grad*lr

        by -= by.grad*lr
from sklearn.metrics import classification_report
result = None

with torch.no_grad():

    for xx, yy in test_loader:

        outputm = xx.mm(wm) + bm

        outputy = (outputm.mm(wy) + by).log_softmax(1)

        if result is None:

            result = outputy.argmax(1)

        else:

            result = torch.cat((result, outputy.argmax(1)))

        

print(classification_report( y_test_tensor.tolist(), result.tolist()))