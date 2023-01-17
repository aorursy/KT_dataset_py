# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch.utils.data import DataLoader,TensorDataset



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


## load the dataset

def load_data():

    train_data = torch.from_numpy(np.array(pd.read_csv('/kaggle/input/digit-recognizer/train.csv'))[:,1:] / 255).view(-1,28,28).float()

    # train_data.size()  # 42000*28*28

    train_label = torch.from_numpy(np.array(pd.read_csv('/kaggle/input/digit-recognizer/train.csv'))[:,0:1]).long()

    # train_label.size() # 42000*1

    train_dataset = TensorDataset(train_data,train_label)

    test_data = torch.from_numpy(np.array(pd.read_csv('/kaggle/input/digit-recognizer/test.csv')) / 255).view(-1,28,28).float()

    # print(test_data.size())  #28000*28*28

    test_dataset = TensorDataset(test_data)

    train_loader = DataLoader(dataset= train_dataset,batch_size = 32,shuffle= True , num_workers= 4)

    test_loader = DataLoader(dataset= test_dataset,batch_size = 32,shuffle= False , num_workers= 4)

    return train_loader, test_loader

# load_data()
import torch.nn as nn

import torch.nn.functional as F

class Minnet(nn.Module):

    def __init__(self):

        super(Minnet,self).__init__()

        self.conv = nn.Sequential(

            #B*1*28*28

            nn.Conv2d(in_channels = 1 , out_channels = 6 , kernel_size = (5,5), stride = 1,padding = 0),

            nn.Dropout(0.5),

            nn.BatchNorm2d(6),

            nn.ReLU(),

            #B*6*24*24

            nn.AvgPool2d(kernel_size = (2,2),stride = 2),

            #B * 6*12*12

            nn.Conv2d(in_channels= 6,out_channels=12, kernel_size=(5,5) ,stride= 1 ,padding = 0),

            nn.Dropout(0.5),

            # B * 12*8*8

            nn.BatchNorm2d(12),

            nn.ReLU(),

            nn.AvgPool2d(kernel_size = (2,2),stride = 2),

            # B * 12* 4*4

        )

        self.fc1 = nn.Linear(in_features= 12*4*4,out_features= 64)

        self.fc2 = nn.Linear(in_features= 64 , out_features= 10)

        

    def forward(self,x):

        # B * 28 * 28

        x = x.unsqueeze(1) #第二个维度上扩充一个维度

        # B * 1 * 28 * 28

        x = self.conv(x)

        # B * 12 * 4 * 4

        x = x.view(x.size(0),-1 )

        #B * (12*4*4)

        x =self.fc1(x)

        x = self.fc2(x)

#         x = F.relu(x)

        # B *  64

#         x = self.fc2(x)

        # B * 10

#         x = F.softmax(x)

        # B * 10 , 每个维度的值在 （0，1） 之间

        return x
from tensorboardX import SummaryWriter

# chkpt_dir = '/kaggle/output/kaggle/working/model'

writer = SummaryWriter(comment = "Minnet")

train_loader ,test_loader = load_data() # load the data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GPU

# print(torch.cuda.is_available())

epoches = 200 # 迭代次数

# criterion = nn.MSELoss().to(device) # loss function

criterion = nn.CrossEntropyLoss().to(device)

model = Minnet().to(device) # load the model

optimizer = torch.optim.Adam(model.parameters(),

                             lr = 0.001 )

model.train()

# optimizer function

# WE need to cal the accuracy and the loss

step_all = len(train_loader)

for epoch in range(epoches):

    total_loss = 0

    num = 0

    rightnum = 0

    for i,data in enumerate(train_loader):

#         print(i)

        x_data,x_label = data[0].to(device),data[1].to(device)

        # x_data. B * 28 *28

        B_size = x_data.size(0) # number of this batch data 

        optimizer.zero_grad()

        out = model(x_data)

#         print(out)

#         print(out)

#         print(x_label)

        # cross entropy 要求target 一定是N维，不能是N *1

        x_label = x_label.squeeze(1)

        loss = criterion(out,x_label)  #cal the loss value

        loss.backward() # back propagation

        optimizer.step()

        out = F.softmax(out)

#         print(out)

        predict = torch.max(out,1)[1]  #找出每行中最大的那一列，并输出索引 # 1* 32

#         print(predict)

#         print(x_label)

        predict = predict.view(predict.size(0),-1).float() # 转换成32 * 1

#         predict.requires_grad = True # 允许计算提督

#         loss = criterion(predict,x_label) #cal the loss value

        

        # cal the total loss for 1 epoch

        total_loss += loss.item() * B_size

        num += B_size

        x_label = x_label.unsqueeze(1)

        # cal the total acc for 1 epoch

        rightnum += x_label.eq(predict.data).cpu().sum().item()

        

        if (i+1) % 100 == 0:

            print("epoch: %d , step: %d/%d , loss: %.5f , acc: %.5f" % (epoch , i+1, step_all , total_loss / num, rightnum / num))

    writer.add_scalar('Train Loss', total_loss/num, epoch)

    writer.add_scalar('Train Accuracy', rightnum/num, epoch)

    print(rightnum , num)

# save the model

# if not os.path.exists(chkpt_dir):

#     os.mkdir(chkpt_dir)

save_path = 'chkpt_%d.pt'%(epoches)

torch.save({

    'model': model.state_dict(),

    'optimizer': optimizer.state_dict(),

    'epoch': epoch,

}, save_path)
model.eval

numtest = 0

res = np.array([])

for i , data in enumerate(test_loader):

    y_data = data[0].to(device)

    numtest+= y_data.size(0)

    out=model(y_data)

    out = F.softmax(out)

    pred = torch.max(out,1)[1]

    res = np.append(np.array(res),np.array(pred.cpu()))

res = res.reshape(res.shape[0],-1)

ind = np.array([i+1 for i in range(res.shape[0])]).reshape(res.shape[0],-1).astype("Int32")

res = np.concatenate((ind,res),axis = 1).astype("Int32")

print(ind.dtype)

print(res.dtype)

# print(res.shape)

dfres = pd.DataFrame(res,columns = ['ImageId','Label'])

dfres.to_csv('minist_submission.csv',index = None)