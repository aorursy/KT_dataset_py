# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/exl-task/train_data_modified21_02.xlsx")

df.head()
labels = df.self_service_platform

labels.head()
df_null = df.isnull()
df_null.sum()
df_var24 = df.var24.copy()
df.var24 = df.var24.fillna(value = df.var24.mean())
df.self_service_platform.value_counts()
df.var37 = df.var37.fillna(value = 0)
df_var38 = df.var38

df = df.drop("var38", axis=1)
dfl = df.self_service_platform

dfl.head()

dfc = df.iloc[[1, 2]].values

dfc[0]

t = torch.ones([200, 2])

t[[1, 2]]
df.var39 = df.var39.fillna(value = 100)
df.columns.shape

# df = df.drop(['Unnamed: 0', 'cust_id'], axis=1)

df.to_excel("final_train_data.xlsx")
print(df.iloc[0, 35])

# print(df.loc[0])
import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
class EXLdataset(Dataset):

    

    def __init__(self, dataframe = None, train = True):

        self.df = dataframe

        self.train = train

        df_labels = self.df.self_service_platform

        self.df = self.df.drop("self_service_platform", axis=1)

        self.df = self.df.apply(lambda x: (x-x.mean())/(x.max()+0.001), axis=0)

        label_tensor = torch.zeros([len(df_labels), 4], dtype=torch.int32)

        for i, a in enumerate(df_labels):

            if a == "Desktop":

                label_tensor[i][0] = 1

            elif a == "Mobile App":

                label_tensor[i][1] = 1

            elif a == "Mobile Web":

                label_tensor[i][2] = 1

            else:

                label_tensor[i][3] = 1

                

        self.labels = label_tensor

        self.df_labels = df_labels

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

            

        data = torch.tensor(self.df.iloc[idx].values)

        label = self.labels[idx]

        

        return {"data":data, "label":label, "actual": self.df_labels.iloc[idx]}    

        

        

        
print(df.columns)

# dfn = df.drop("self_service_platform", axis=1)

print(dfn.columns)



dfn = df.sample(frac=1)

df_train = dfn.iloc[[a for a in range(250000)]]

df_val = dfn.iloc[[a for a in range(250000, 300000)]]

my_data_train = EXLdataset(dataframe = df_train, train = True)

my_data_val = EXLdataset(dataframe = df_val, train = True)



# for i in range(len(my_data)):

#     sample = my_data[i]

#     print(sample["data"].shape, sample["label"].shape)

#     print(sample["data"])

#     if i>1:

#         break
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(37, 2048) 

        self.fc2 = nn.Linear(2048, 512)

        self.fc3 = nn.Linear(512, 128)

        self.fc4 = nn.Linear(128, 64)

        self.fc5 = nn.Linear(64, 32)

        self.fc6 = nn.Linear(32, 16)

        self.fc7 = nn.Linear(16, 8)

        self.fc8 = nn.Linear(8, 4)



    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))

        x = F.leaky_relu(self.fc2(x))

        x = F.leaky_relu(self.fc3(x))

        x = F.leaky_relu(self.fc4(x))

        x = F.leaky_relu(self.fc5(x))

        x = F.leaky_relu(self.fc6(x))

        x = F.leaky_relu(self.fc7(x))

        x = self.fc8(x)

        return x



    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features





net = Net()

print(net)
dataloader_train = DataLoader(my_data_train, batch_size = 256, shuffle = True, num_workers = 4)

dataloader_val = DataLoader(my_data_val, batch_size = 256, shuffle = False, num_workers = 4)
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(net.parameters(), lr = 0.001, weight_decay = 0.001, momentum=0.9)
running_loss = 0.0

train_loss = []

val_loss = []

for epoch in range(200):

    running_loss = 0.0

    for i, sample in enumerate(dataloader_train, 0):

        inputs = sample["data"]

        labels = sample["label"]

        df_labels = sample["actual"]

        optimizer.zero_grad()

        outputs = net(inputs.float())

        labels = labels.type_as(outputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

#         print(outputs.shape)

        new = torch.zeros(outputs.shape)

        for a in range(outputs.shape[0]):

            new[a][outputs[a]==outputs[a].max()] =1

            

        acc=0.0

        for a in range(outputs.shape[0]):

            if torch.equal(new[a], labels[a]):

                acc = acc+1

            

        acc = acc/256

        

#         if i<3:

#             print(labels, df_labels)

        

        running_loss += loss.item()

        

        if i%100 == 0:

            print("epoch ", epoch, " i " , i, " loss ", loss.item(), " Accuracy ", acc)

        if i%10==1:

            train_loss.append(loss.item())

            

    running_loss = 0.0    

    for i, sample in enumerate(dataloader_val, 0):

        inputs = sample["data"]

        labels = sample["label"]

#         optimizer.zero_grad()

        outputs = net(inputs.float())

        labels = labels.type_as(outputs)

        loss = criterion(outputs, labels)

        

        running_loss += loss.item()

        

        new = torch.zeros(outputs.shape)

        for a in range(outputs.shape[0]):

            new[a][outputs[a]==outputs[a].max()] =1

            

        acc=0.0

        for a in range(outputs.shape[0]):

            if torch.equal(new[a], labels[a]):

                acc = acc+1

            

        acc = acc/256

        

#         if i%100 == 0:

#             print("epoch ", epoch, " i " , i, " loss ", loss.item())

        if i%5==1:

            print("val_loss ", loss.item(), " Accuracy ", acc)

            val_loss.append(loss.item())

#             val_loss.append(running_loss/i)

        
import matplotlib.pyplot as plt

plt.plot(train_loss)

plt.plot(val_loss)
plt.plot(val_loss)
t = torch.randn([32, 4], dtype = torch.float64)
new = torch.zeros(t.shape)
for i in range(32):

    new[i][t[i]==t[i].max()] =1
new.shape[0]
t = new
new[0][3]=1
t[0]

for a in range(32):

    print(torch.equal(t[a], torch.tensor([1, 0, 0, 0]).float()))
torch.save(net.state_dict(), "/kaggle/working/weights2")