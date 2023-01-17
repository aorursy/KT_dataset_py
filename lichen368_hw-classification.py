# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import torch

import torch.nn as nn

import torch.nn.functional as fn

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
n_data=torch.ones(100,2)

x0=torch.normal(2*n_data,1)

y0=torch.zeros(100)

x1=torch.normal(-2*n_data,1)

y1=torch.ones(100)

x=torch.cat((x0,x1),dim=0).type(torch.FloatTensor)

y=torch.cat((y0,y1,)),type(torch.LongTensor)

print(x)


class Classification(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):

        super(Classification,self).__init__()

        self.fc1=nn.Linear(n_input,n_hidden)

        self.fc2=nn.Linear(n_hidden,n_hidden)

        self.fc3=nn.Linear(n_hidden,n_output)

        self.relu=nn.ReLU()

    def forward(self,x):

        x=self.fc1(x)

        x=self.relu(x)

        x=self.fc2(x)

        x=self.relu(x)

        x=self.fc3(x)

        return x
cls_net=Classification(2,10,2)

optimizer=torch.optim.SGD(cls_net.parameters(),lr=0.02)

loss_func = torch.nn.CrossEntropyLoss()

for i in range(2):

    pred=cls_net(x)

    print(x)

    loss_value=loss_func(pred,y)

   # optimizer.zero_grad()

#     loss_value.backward()

#     optimizer.step()

#     print("loss")

#     print(loss_value)

#     if i%5 ==0:

#         plt.cla()

#         prediction=torch.max(pred,1)[1]

#         print(pred[0:2])

#         print(torch.max(pred,1)[0:2])

#         print(torch.max(pred,1)[1])

#         predy=prediction.data.numpy()

#         targety=y.data.numpy()

#         print("y1")

#         print(predy)

#         print("y2")

#         print(targety)

#         plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=predy,s=100,lw=0,cmap="RdYlGn")

#         accuracy=float((predy==targety).astype(int).sum()/float(targety.size))

#         plt.text(1.5,-4,"Accuracy=={:.2}".format(accuracy),fontdict={"size":20,"color":"red"})

#         plt.pause(0.1)

    

#     plt.ioff()

#     plt.show()

    
a=torch.randn(3,3)

print(a)

torch.max(a,1)[1]


