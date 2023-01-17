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
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import pandas as pd
class OttoDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('../input/train.csv',delimiter=',',skiprows = 1, usecols = np.arange(1,94))
        df = pd.read_csv('../input/train.csv', sep = ',')
        df['target'] =  df['target'].map({'Class_1': 1, 'Class_2': 2,
                                          'Class_3': 3, 'Class_4': 4,
                                          'Class_5': 5, 'Class_6': 6,
                                          'Class_7': 7, 'Class_8': 8,
                                          'Class_9': 9})
        df['target'] = df['target'].astype('float64')
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:])
        self.y_data = torch.tensor(df['target'].values)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
dataset = OttoDataset()

train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(93, 46)
        self.l2 = nn.Linear(46,18)
        self.l3 = nn.Linear(18,9)
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
loss = 1000
for epoch in range(10):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = Variable(data).float(),Variable(target).type(torch.LongTensor)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target-1)
        loss.backward()
        optimizer.step()
  #if batch_idx % 10 == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
              #  100. * batch_idx / len(train_loader), loss.data[0]))
        #print(data.shape)
        #print(target.shape)
        #print(epoch, i, "inputs", data, "\n labels", target)
#loss.item()
model.eval()
xyTest = np.loadtxt('../input/test.csv',delimiter=',',skiprows = 1, usecols = np.arange(1,94))
df1 = pd.read_csv('../input/test.csv',sep=',')
#xyTest.shape
xy_pred = torch.from_numpy(xyTest[:,:])
type(xy_pred)
id_col = df1['id']
class_list = ['id','Class_1','Class_2','Class_3','Class_4','Class_5',
             'Class_6','Class_7','Class_8','Class_9']
class_list2 = ['Class_1','Class_2','Class_3','Class_4','Class_5',
             'Class_6','Class_7','Class_8','Class_9']
d = pd.DataFrame(0, index=np.arange(xy_pred.shape[0]), columns=class_list)
d['id'] = df1['id']
d[class_list2] = d[class_list2].astype('float')
d.dtypes
d.head()
classify = 'Class_'
#print(df1.iloc[2,1:])
for i in range(xy_pred.shape[0]):
    output = model(Variable(xy_pred[i]).float())
    row = F.softmax(output).data
    classes = row.numpy()
    classes = np.around(classes, decimals=1)
    print(type(classes))
    print(classes)
    d.loc[i,1:] = classes
    #print(indices.item()+1)
    #no = indices.item()+1
    #pred = classify + str(no)
    #d.iloc[i][pred] = 1
d.head()
d.to_csv('submission1.csv',index = False)







