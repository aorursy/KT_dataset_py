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
import numpy as np

import pandas as pd

# read data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

print("Load data sucessfully!")
print("Begin Data Processing")

# understand data

print(train_data.head())

print(test_data.head())
# convert pd to np

train_data = train_data.values

test_data = test_data.values

print(train_data.shape)

print(test_data.shape)
# split to X,y,  and norm X

X_train = train_data[:,1:].reshape(-1,1,28,28)/255

y_train = train_data[:,0:1]
# # one-hot encoder

# from sklearn.preprocessing import OneHotEncoder

# #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

# onehot_encoder = OneHotEncoder(sparse=False)

# y_train = onehot_encoder.fit_transform(y_train)
# split to train valid

from sklearn.model_selection import train_test_split

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,

                                                      test_size=0.2, random_state=2019)
print(X_train.shape)

print(X_valid.shape)

print(y_train.shape)

print(y_valid.shape)

print("Data Processing Successful!")
# #https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# # Data augmentation and normalization for training

# # Just normalization for validation

# data_transforms = {

#     'train': transforms.Compose([

#         transforms.RandomResizedCrop(224),

#         transforms.RandomHorizontalFlip(),

#         transforms.ToTensor(),

#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#     ]),

#     'val': transforms.Compose([

#         transforms.Resize(256),

#         transforms.CenterCrop(224),

#         transforms.ToTensor(),

#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#     ]),

# }



# data_dir = 'data/hymenoptera_data'

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

#                                           data_transforms[x])

#                   for x in ['train', 'val']}

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,

#                                              shuffle=True, num_workers=4)

#               for x in ['train', 'val']}

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# class_names = image_datasets['train'].classes



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model parameters

X_channel = 1

filter1, kernel1, padding1, max_pooling1 = 16, 3, 1, 2

filter2, kernel2, padding2, max_pooling2 = 32, 3, 1, 2

filter3, kernel3, padding3, max_pooling3 = 64, 3, 1, 2

dense0, dense1, dense2, dense3 = 64*3*3, 120, 64, 10
import torch

import torch.nn as nn

import torch.nn.functional as F

#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        # convolution layers

        self.conv1 = nn.Conv2d(X_channel, filter1, kernel1, padding=padding1) 

        self.conv2 = nn.Conv2d(filter1, filter2, kernel2, padding=padding2) 

        self.conv3 = nn.Conv2d(filter2, filter3, kernel3, padding=padding3) 

        

        # fully connect

        self.fc1 = nn.Linear(dense0, dense1)

        self.fc2 = nn.Linear(dense1, dense2)

        self.fc3 = nn.Linear(dense2, dense3)

        

    def forward(self, X):

        X = F.max_pool2d(F.relu(self.conv1(X)), max_pooling1)

        X = F.max_pool2d(F.relu(self.conv2(X)), max_pooling2)

        X = F.max_pool2d(F.relu(self.conv3(X)), max_pooling3)

        X = X.view(-1,dense0)

        X = F.relu(self.fc1(X))

        X = F.relu(self.fc2(X))

        X = self.fc3(X)

        return X



net = Net()

print(net)
# set train parameters

# https://github.com/zergtant/pytorch-handbook/blob/master/chapter3/3.2-mnist.ipynb

batch_size = 512 #大概需要2G的显存

EPOCH = 200 # 总共训练批次

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

print(device)
import torch.utils.data
# Pytorch train and test sets

#https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers/notebook

train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),

                                       torch.from_numpy(y_train).long())

# https://blog.csdn.net/baidu_36639782/article/details/86641866

valid = torch.utils.data.TensorDataset(torch.from_numpy(X_valid).float(),

                                      torch.from_numpy(y_valid).long())



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

valid_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle = False)
def train(model, device, train_loader, optimizer, criterion):

    model.train()#把module设成training模式，对Dropout和BatchNorm有影响

#     best_model_wts = copy.deepcopy(model.state_dict())

#     best_acc = 0.0

    running_loss = 0.0

    running_corrects = 0            

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        #https://stackoverflow.com/questions/49206550/ ...

        #...   pytorch-error-multi-target-not-supported-in-crossentropyloss

        preds = torch.argmax(output, 1)

        loss = criterion(output, target.squeeze_())

        loss.backward()

        optimizer.step()

        # statistics

        running_loss += loss.item() * data.size(0)

        running_corrects += torch.sum(preds == target)

    epoch_loss = running_loss / len(X_train)

    epoch_acc = running_corrects.double() / len(X_train)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format("Training", epoch_loss, epoch_acc))
def valid(model, device, test_loader, criterion):

    model.eval()#把module设置为评估模式，只对Dropout和BatchNorm模块有影响

#     best_model_wts = copy.deepcopy(model.state_dict())

#     best_acc = 0.0

    running_loss = 0.0

    running_corrects = 0     

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            preds = torch.argmax(output, 1)

            loss = criterion(output, target.squeeze_())

            running_loss += loss.item() * data.size(0)

            running_corrects += torch.sum(preds == target)

    epoch_loss = running_loss / len(X_valid)

    epoch_acc = running_corrects.double() / len(X_valid)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format("valid", epoch_loss, epoch_acc))

    return epoch_acc
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

model = net.to(device)

import time,copy

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())

best_acc = 0.0



for epoch in range(1, EPOCH + 1):



    print('Epoch {}/{}:'.format(epoch, EPOCH))

    print('-' * 10)

    train(model, device, train_loader, optimizer, criterion)

    epoch_acc = valid(model, device, valid_loader, criterion)

    if best_acc<epoch_acc:

        best_acc = epoch_acc

        best_model_wts = copy.deepcopy(model.state_dict())

    else:

        print("best acc is {:.4f}".format(best_acc))

time_elapsed = time.time() - since

print('Training complete in {:.0f}m {:.0f}s'.format(

    time_elapsed // 60, time_elapsed % 60))

print('Best val Acc: {:4f}'.format(best_acc))
model.load_state_dict(best_model_wts)
X_test = test_data.reshape(-1,1,28,28)/255

test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float())



# data loader

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
def test(model, device, test_loader, criterion):

    model.eval()#把module设置为评估模式，只对Dropout和BatchNorm模块有影响

#     best_model_wts = copy.deepcopy(model.state_dict())

#     best_acc = 0.0

    ans = []

    with torch.no_grad():

        for data in test_loader:

            data = data[0].to(device)

            output = model(data)

            preds = torch.argmax(output, 1)

            ans.append(preds)

    return ans
Ans = test(model, device, test_loader, criterion)
Ans = torch.cat(Ans).cpu().numpy()
Ans
res = pd.read_csv("../input/sample_submission.csv")
res["Label"] = Ans
res.to_csv("res.csv",index=False)