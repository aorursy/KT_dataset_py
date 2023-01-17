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
from __future__ import absolute_import

from __future__ import division

import numpy as np

import scipy.io as sio

import os

import torch

import torchvision

import torch.nn as nn

import math

from torch.optim.lr_scheduler import StepLR

import random

from torch.autograd import Variable

import torch.nn.functional as F

from torch.nn.modules.conv import _ConvNd

# 各种卷积函数

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# from mat to matrix

def file2matrix(filename):  

    mat_data = sio.loadmat(filename)

    data = mat_data['Qt'][0]



    # 训练数据样本数train_num、测试数据样本数test_num、每个样本的采样点数time_num、类别数

    data_num = len(data)

    time_num = len(data[0][0])

    class_num = int(data[-1][0][0][1]) + 1



    X_set = np.zeros([data_num, time_num, 1], dtype='float32')

    y_set = np.zeros([data_num, 1], dtype='int32')



    for i in range(data_num):

        for j in range(time_num):

            X_set[i][j][0] = data[i][0][j][0]

        y_set[i][0] = data[i][0][0][1]



    

    return X_set,y_set,data_num,time_num,class_num
filename_train = '/kaggle/input/modultaion-recognition/train-20sample-20db.mat'

metatrain_folders,metatrain_y_folders,metatrain_num,metatrain_time,metatrain_class = file2matrix(filename_train)
filename_test = '/kaggle/input/modultaion-recognition/test-20db.mat'

metatest_folders,metatest_y_folders,metatest_num,metatest_time,metatrain_class = file2matrix(filename_test)

print(metatrain_num)

print(metatest_num)
# 初始化权重

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        n = m.kernel_size[0] * m.out_channels

        m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('BatchNorm') != -1:

        m.weight.data.fill_(1)

        m.bias.data.zero_()

    elif classname.find('Linear') != -1:

        n = m.weight.size(1)

        m.weight.data.normal_(0, 0.01)
class ResNet(nn.Module):

# 9层Res+GAP+Softmax 分成11类

    def __init__(self):

        # 定义卷积网络的结构

        super(ResNet, self).__init__()

        self.layer1 = nn.Sequential(

                        nn.Conv1d(in_channels=1,out_channels=8,kernel_size=9,padding = 4),

                        nn.BatchNorm1d(8,eps=1e-05, momentum=0.1, affine=True),

                        nn.ReLU(),

                        nn.Conv1d(in_channels=8,out_channels=8,kernel_size=5,padding = 2),

                        nn.BatchNorm1d(8,eps=1e-05, momentum=0.1, affine=True),

                        nn.ReLU(),

                        nn.Conv1d(in_channels=8,out_channels=8,kernel_size=3,padding = 1),

                        nn.BatchNorm1d(8,eps=1e-05, momentum=0.1, affine=True))

        

        self.layer2 = nn.Sequential(

                        nn.Conv1d(in_channels=8,out_channels=16,kernel_size=9,padding = 4),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True),

                        nn.ReLU(),

                        nn.Conv1d(in_channels=16,out_channels=16,kernel_size=5,padding = 2),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True),

                        nn.ReLU(),

                        nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding = 1),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True))

        self.layer3 = nn.Sequential(

                        nn.Conv1d(in_channels=16,out_channels=16,kernel_size=9,padding = 4),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True),

                        nn.ReLU(),

                        nn.Conv1d(in_channels=16,out_channels=16,kernel_size=5,padding = 2),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True),

                        nn.ReLU(),

                        nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding = 1),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True))

        self.layer4 = nn.AdaptiveAvgPool1d((11))

        self.layer5 = nn.Linear(16*11,11)

        self.shortcut1 = nn.Sequential(

                        nn.Conv1d(in_channels=1,out_channels=8,kernel_size=1),

                        nn.BatchNorm1d(8,eps=1e-05, momentum=0.1, affine=True))

        self.shortcut2 = nn.Sequential(

                        nn.Conv1d(in_channels=8,out_channels=16,kernel_size=1),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True))

        self.shortcut3 = nn.Sequential(

                        nn.Conv1d(in_channels=16,out_channels=16,kernel_size=1),

                        nn.BatchNorm1d(16,eps=1e-05, momentum=0.1, affine=True))

    def forward(self,x):

        y = self.layer1(x)

        shortcut_y = self.shortcut1(x)

        y = torch.add(shortcut_y, y)

        y = F.relu(y)

        z = self.layer2(y)

        shortcut_z = self.shortcut2(y)

        z = torch.add(shortcut_z, z)

        z = F.relu(z)

        out = self.layer3(z)

        shortcut_out = self.shortcut3(z)

        out = torch.add(out, shortcut_out)

        out = F.relu(out)

        out = self.layer4(out)

        out = out.view(out.size(0),-1)

        out = self.layer5(out)

        out = F.softmax(out)

        return out
LEARNING_RATE = 0.001

GPU = 0



resnet = ResNet()

resnet.apply(weights_init)

resnet.cuda(GPU)



resnet_optim = torch.optim.Adam(resnet.parameters(),lr=LEARNING_RATE)

resnet_scheduler = StepLR(resnet_optim,step_size=10000,gamma=0.5)
#抽batch的同时打标签

class generateTask(object):

    def __init__(self, data_folders, label_folders, batch_num, time_num, class_num):



        self.data_folders = data_folders

        self.label_folders = label_folders

        self.batch_num = batch_num

        self.time_num = time_num

        self.class_num = class_num

        

        

        # 抽取的内容

        self.sample_data = np.zeros((batch_num,time_num,1))

        self.sample_label = np.zeros((batch_num,class_num))

        self.real_label = np.zeros((batch_num))



        

        #不能破坏原有的数据结构,所以只产生索引

        idx = np.random.choice(len(data_folders),batch_num,replace = False)

        #打标签的时候直接one-hot

        for i in range(batch_num):

            self.sample_data[i] = self.data_folders[idx[i]]

            tmp_label = self.label_folders[idx[i]]

            self.real_label[i] = tmp_label

            self.sample_label[i][tmp_label] = 1

        

        self.sample_data = self.sample_data.reshape(batch_num,1,1500)

        self.sample_data = (torch.from_numpy(self.sample_data)).float()

        self.sample_label = (torch.from_numpy(self.sample_label)).float()
BATCH_NUM = 32

CLASS_NUM = 10

EPISODE = 2000

accuracy = 0.0

TEST_EPISODE = 1000

for episode in range(EPISODE):

    resnet_scheduler.step(episode)

    #1:生成task

    task = generateTask(metatrain_folders,metatrain_y_folders,BATCH_NUM,metatrain_time,metatrain_class)

    data,label = task.sample_data,task.sample_label

    scores = resnet(Variable(data).cuda(GPU))



    #print(scores.shape) 128*11

    mse = nn.MSELoss().cuda(GPU)

    label = Variable(label).cuda(GPU)

    loss = mse(scores,label)

    # 找到每行最大值

    #_,predict_labels = torch.max(relations.data,1)

    

    # training

    resnet.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(resnet.parameters(),0.5)

    resnet_optim.step()

    

    #每100step打印loss信息

    if (episode+1)%100 == 0:

            print("episode:",episode+1,"loss",loss.item())

            

    if (episode+1)%1000 == 0:

        total_rewards = 0

        confusion = np.zeros((metatrain_class,metatrain_class))

        for i in range(TEST_EPISODE):

            task = generateTask(metatest_folders,metatest_y_folders,BATCH_NUM,metatest_time,metatrain_class)

            data,label = task.sample_data,task.sample_label

            real_label = task.real_label

            scores = resnet(Variable(data).cuda(GPU))



            _,predict_labels = torch.max(scores.data,1)

            for i in range(BATCH_NUM):

                confusion[int(real_label[i])][int(predict_labels[i])] += 1

                if predict_labels[i] == real_label[i]:

                    total_rewards += 1

                else:

                    total_rewards = total_rewards



        #print(total_rewards)

        #验证测试精度并打印

        test_accuracy = total_rewards/1.0000/BATCH_NUM/TEST_EPISODE

        print("test accuracy:",test_accuracy)

        

        if(test_accuracy > accuracy):

            accuracy = test_accuracy

            torch.save(resnet.state_dict(), 'model.pkl')

        else:

            accuracy = accuracy