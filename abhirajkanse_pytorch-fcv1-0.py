# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from __future__ import print_function, division

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import os

import torch

from skimage import io, transform

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

print(train_features)
class TrainDataset(Dataset):



    def __init__(self, data_csv_file, target_csv_file, transform=None):

        self.train_features = pd.read_csv(data_csv_file)

        self.targets = pd.read_csv(target_csv_file)

        cp_type = {'trt_cp': 1,'ctl_vehicle': 0}

        cp_dose = {'D1': 1,'D2': 0}

        cp_time = {24 : 1.0/3 , 48: 2.0/3 , 72: 1}

        self.train_features.cp_type = [cp_type[item] for item in self.train_features.cp_type]

        self.train_features.cp_dose = [cp_dose[item] for item in self.train_features.cp_dose]

        self.train_features.cp_time = [cp_time[item] for item in self.train_features.cp_time]

        self.transform = transform



    def __len__(self):

        return len(self.train_features)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        train_features = self.train_features.iloc[idx, 1:]

        targets = self.targets.iloc[idx, 1:]

        train_features = np.array([train_features])

        targets = np.array([targets])

        train_features = train_features.astype('float').reshape(-1)

        targets = targets.astype('float').reshape(-1)

        sample = {'train_features': train_features, 'targets' : targets}



        if self.transform:

            sample = self.transform(sample)



        return sample

    

class TestDataset(Dataset):



    def __init__(self, data_csv_file, transform=None):

        self.test_features = pd.read_csv(data_csv_file)

        cp_type = {'trt_cp': 1,'ctl_vehicle': 0}

        cp_dose = {'D1': 1,'D2': 0}

        cp_time = {24 : 1.0/3 , 48: 2.0/3 , 72: 1}

        self.test_features.cp_type = [cp_type[item] for item in self.test_features.cp_type]

        self.test_features.cp_dose = [cp_dose[item] for item in self.test_features.cp_dose]

        self.test_features.cp_time = [cp_time[item] for item in self.test_features.cp_time]

        self.transform = transform



    def __len__(self):

        return len(self.test_features)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        test_features = self.test_features.iloc[idx, 1:]

        test_features = np.array([test_features])

        test_features = test_features.astype('float').reshape(-1)

        sample = {'test_features': test_features}



        if self.transform:

            sample = self.transform(sample)



        return sample
train_dataset = TrainDataset(data_csv_file='/kaggle/input/lish-moa/train_features.csv', target_csv_file = '/kaggle/input/lish-moa/train_targets_scored.csv')

sample = train_dataset[0]

#print(sample['train_features'].shape)

trainloader = DataLoader(train_dataset, batch_size=64,shuffle=True, num_workers=0)

for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data['train_features'], data['targets']

        print(torch.max(labels, 1)[1])

        break
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self, in_D, h_D ,out_D):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_D, h_D[0])

        self.bn1 = nn.BatchNorm1d(h_D[0])

        self.do1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(h_D[0], h_D[1])

        self.bn2 = nn.BatchNorm1d(h_D[1])

        self.do2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(h_D[1], h_D[2])

        self.bn3 = nn.BatchNorm1d(h_D[2])

        self.fc4 = nn.Linear(h_D[2], out_D)



    def forward(self, x):

        x = F.relu(self.do1(self.bn1(self.fc1(x))))

        x = F.relu(self.do2(self.bn2(self.fc2(x))))

        x = F.relu(self.do2(self.bn3(self.fc3(x))))

        x = self.fc4(x)

        return x



in_D = 875

h_D = [2048 , 1024, 512]

out_D = 206



def init_weights(m):

    if type(m) == nn.Linear:

        torch.nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))

        m.bias.data.fill_(0.01)



def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    model = checkpoint['model']

    model.load_state_dict(checkpoint['state_dict'])

    for parameter in model.parameters():

        parameter.requires_grad = False



    model.eval()

    return model



net = Net(in_D,h_D,out_D)

try:

    net = load_checkpoint('checkpoint_CE.pth')

except:

    net.apply(init_weights)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9, weight_decay=0.01)

schedular = optim.lr_scheduler.MultiStepLR(optimizer, [10,20,30], gamma=0.1, last_epoch=-1)
import time

start_t = time.time()

loss_arr = []

for epoch in range(40):  # loop over the dataset multiple times

    

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data['train_features'], data['targets']



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs.float())

        #print(labels.shape)

        loss = criterion(outputs, torch.max(labels, 1)[1])

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 100 == 99: 

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 100))

            loss_arr.append(running_loss/100)

            running_loss = 0.0

    schedular.step()

            

print('Finished Training')

end_t = time.time()

print("Time taken : ",end_t-start_t)

plt.plot(loss_arr)

train_dataset = TrainDataset(data_csv_file='/kaggle/input/lish-moa/train_features.csv', target_csv_file = '/kaggle/input/lish-moa/train_targets_scored.csv')

trainloader = DataLoader(train_dataset, batch_size=64,shuffle=True, num_workers=0)

running_sum = 0

num_inst = 0

with torch.no_grad():

    for data in trainloader:

        feat, labels = data['train_features'], data['targets'] 

        outputs = model(feat.float())

        #print(torch.argmax(outputs, dim =1))

        #print(torch.max(labels, 1)[1])

        #print(torch.max(labels, 1)[1] - torch.argmax(outputs, dim =1))

        #print(torch.sum(torch.argmax(outputs, dim =1) == torch.max(labels, 1)[1]))

        running_sum += torch.sum(torch.argmax(outputs, dim =1) == torch.max(labels, 1)[1])

        num_inst += feat.shape[0]

print('acc : ', float(running_sum)/num_inst)
checkpoint = {'model': Net(in_D,h_D,out_D),

          'state_dict': net.state_dict(),

          'optimizer' : optimizer.state_dict()}



torch.save(checkpoint, 'checkpoint_CE.pth')



model = load_checkpoint('checkpoint_CE.pth')
test_dataset = TestDataset(data_csv_file='/kaggle/input/lish-moa/test_features.csv')

testloader = DataLoader(test_dataset, batch_size=64,shuffle=False, num_workers=0)        
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

pointer = 0

with torch.no_grad():

    for data in testloader:

        feat = data['test_features']

        outputs = model(feat.float())

        outputs = outputs.numpy()

        temp = np.zeros_like(outputs)

        args = np.argmax(outputs, axis =1)

        for i in range(outputs.shape[0]):

            temp[i][args[i]] = 1

        for i in range(outputs.shape[0]):

            submission.iloc[pointer+i,1:] = temp[i]

        pointer += 64
print(submission)
submission.to_csv('submission.csv',mode = 'w', index=False)