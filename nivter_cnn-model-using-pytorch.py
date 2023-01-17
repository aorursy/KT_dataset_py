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
from torch.utils.data import Dataset
from torchvision import transforms
import glob, os
from PIL import Image

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
# Store labels in dict
with open('../input/train_responses.csv') as f:
    labels = f.read().strip().split('\n')
    labels = [i.split(',') for i in labels]
    labels = [[i[0], float(i[1])] for i in labels[1:]]
    labels = dict(labels)
    
len(labels.keys())
# Define loader class
class Loader(Dataset):
    def __init__(self, data_dir, transform):
        self.images = glob.glob(os.path.join(data_dir, '*.png'))
        self.transform = transform
        print('Length of {}: {}'.format(data_dir, len(self.images)))

    def __getitem__(self, index):
        image = self.images[index]
        k = os.path.basename(image).split('.')[0]
        image = Image.open(image).convert('L')
        image = self.transform(image)
        return image, labels[k]

    def __len__(self):
        return len(self.images)
  
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
train_data = Loader('../input/train_imgs/train_imgs/', transform)
len(train_data)
# Set parameter values
BATCH_SIZE = 64
EPOCHS = 3
LR = 0.001
GAMMA = 0.1
use_cuda = torch.cuda.is_available()
criterion = nn.MSELoss()
test_criterion = nn.MSELoss(size_average=False)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.name = 'simple_cnn'
        self.main = nn.Sequential(
            nn.Conv2d(1, 5, 3, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(5),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(5, 10, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(10),

            nn.Conv2d(10, 15, 3),
            nn.ReLU(True),
            nn.BatchNorm2d(15),

            nn.MaxPool2d(2, stride=2)
        )
        self.fc1 = nn.Linear(15*16*16, 500)
        self.fc2 = nn.Linear(500, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tanh(x)
        return x

model_simple = SimpleCNN()
if use_cuda: model_simple.cuda()

train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
optimizer = torch.optim.Adam(model_simple.parameters(), lr = LR)
def to_variable(x):
    if use_cuda: x = x.cuda()
    return Variable(x)

def train(model, epoch):
    model.train()
    print('Learning rate is:', optimizer.param_groups[0]['lr'])
    for i, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target.float())
        target = torch.unsqueeze(target, 1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i%20 == 0:
            print('Epoch [{}], iteration [{}], loss [{}]'.format(epoch, i+1, loss.data[0]))
        if i == 199:
            break # just want to run this for 200 iterations to save time

train(model_simple, 1)