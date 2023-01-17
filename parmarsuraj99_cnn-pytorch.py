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
import cv2
from PIL import Image

from matplotlib import pyplot as plt
import torch
DATA_DIR = '../input/'

TRAIN_DIR = DATA_DIR+'train/train/'

NAMES_DIR = DATA_DIR+'train.csv'
TEST_DIR= '../input/test/test/'
train_names = pd.read_csv(NAMES_DIR)

train_names.shape
len(os.listdir(TRAIN_DIR))
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

import torchvision.transforms as transforms
def show_image(index):

    image = Image.open(TRAIN_DIR+str(train_names['id'][index]))

    plt.imshow(image)

    print(train_names['has_cactus'][index])



    

show_image(20)
class HasCactusDataset(Dataset):

    

    def __init__(self, csv_file, root_dir, transform=None):

    

        self.img_names = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform

    

    def __len__(self):

        return len(self.img_names)



    def __getitem__(self, idx):

        img_name=os.path.join(self.root_dir,

                             self.img_names.iloc[idx, 0])

        image = cv2.imread(img_name)

        image = torch.from_numpy(image)

        

        image = image.permute(2, 0, 1)

        #print(image)

        

        has_cactus = self.img_names.iloc[idx, 1]

        #print(has_cactus)

        sample = [image,has_cactus]

            

        return sample
hasCactusDataset = HasCactusDataset(csv_file=DATA_DIR+'train.csv',

                                   root_dir=TRAIN_DIR,

                                   transform=transforms.Compose([

                                       transforms.ToTensor()

                                   ]))
plt.imshow(hasCactusDataset[19][0].permute(1, 2, 0).numpy())
dataloader = DataLoader(hasCactusDataset, 

                       batch_size=128,

                       shuffle=True)
import torch.nn as nn

import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(10, 18, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(18, 25, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(25, 30, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(30, 40, 5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(40 * 6 * 6, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 1)

        

    def forward(self, x):

        x = F.relu(self.conv1(x))

        #print(x.shape)

        x = F.relu(self.conv2(x))

        #print(x.shape)

        x = self.pool(F.relu(self.conv3(x)))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = self.pool(F.relu(self.conv6(x)))

        print(x.shape)

        x = x.view(-1, 40 * 6 * 6)

        #print(x.shape)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))

        return x

    

net = Net().cuda()
import torch.optim as optim



critation = nn.BCELoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
from torch.autograd import Variable
for epoch in range(10):

    running_loss=0

    print("epoch {} started...", format(epoch))

    for i, data in enumerate(iter(dataloader)):

        image, label = data

        optimizer.zero_grad()

        

        inputs = image.type(torch.FloatTensor)

        inputs=Variable(inputs).cuda()

        label = label.view(-1, 1)

        label = Variable(label).cuda()

        

        outputs = net(inputs)

        loss=critation(outputs.type(torch.FloatTensor),label.type(torch.FloatTensor))

        loss.backward()

        optimizer.step()

        

        

    print("epoch completed", format(epoch))

        

print('finished Trainig...')
data = next(iter(dataloader))
class TestSet(Dataset):

    

    def __init__(self, root_dir, transform=None):

    

        self.img_names = os.listdir(root_dir)

        self.root_dir = root_dir

        self.transform = transform

    

    def __len__(self):

        return len(os.listdir(self.root_dir))



    def __getitem__(self, idx):

        img_loc=os.path.join(self.root_dir,

                             self.img_names[idx])

        img_name = self.img_names[idx]

        #print(img_name)

        image = cv2.imread(img_loc)

        image = torch.from_numpy(image)

        

        image = image.permute(2, 0, 1)

        

        name = img_name

        #print(name)

        sample = [image,name]

            

        return sample
testSet = TestSet(root_dir=TEST_DIR,

                                   transform=transforms.Compose([

                                       transforms.ToTensor()

                                   ]))
testloader = DataLoader(testSet, 

                       batch_size=1,

                       shuffle=True)
tstdata = next(iter(testloader))
tstdata[0].shape
plt.imshow(tstdata[0][0].permute(1,2,0).numpy())
tst_outputs =[]

tst_names = []

for i, data in enumerate(iter(testloader), 0):

    image, name = data[0], data[1]

    #print(name[0])

    inputs = image.type(torch.FloatTensor)

    inputs= Variable(inputs).cuda()

    tst_outputs.append(net(inputs).item())

    tst_names.append(name[0])
len(tst_outputs)
tst_outputs_rounded = np.round(tst_outputs)
(tst_outputs_rounded==1).sum()
len(tst_names)
my_submission = pd.DataFrame({'id': tst_names, 'has_cactus': tst_outputs_rounded})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)