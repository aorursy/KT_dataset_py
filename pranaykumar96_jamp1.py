# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



 # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

import torch, torchvision

from torchvision import transforms

import matplotlib.pyplot as plt

import cv2

from PIL import Image

from torch.utils.data import Dataset, DataLoader, ConcatDataset
train_dir = "/kaggle/input/jamp-hackathon-drive-1/train_set/humans/"

train_imgs = ['/kaggle/input/jamp-hackathon-drive-1/train_set/humans/{}'.format(i) for i in os.listdir(train_dir)]

train_imgs.append(['/kaggle/input/jamp-hackathon-drive-1/train_set/non_human/{}'.format(i) for i in os.listdir('/kaggle/input/jamp-hackathon-drive-1/train_set/non_human/')])
train_dir = "/kaggle/input/jamp-hackathon-drive-1/train_set/"

test_dir = "/kaggle/input/jamp-hackathon-drive-1/test_set/"



train_files_human = os.listdir(train_dir+"humans")

train_files_non_human = os.listdir(train_dir+"non_human")

test_files = os.listdir(test_dir)

class dataset(Dataset):

    def __init__(self, file_list, dir, mode, transform = None):

        self.file_list = file_list

        self.dir = dir

        self.mode = mode

        self.transform = transform

        if self.mode == 'train':

            if 'non_human' in self.file_list[0]:

                self.label = 0

            else:

                self.label = 1

        

    def __len__(self):

        return len(self.file_list)

    

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.dir, self.file_list[idx]))

        if self.transform:

            img = self.transform(img)

        if self.mode == 'train':

            img = img.numpy()

            return img.astype('float32'), self.label

        else:

            img = img.numpy()

            return img.astype('float32'), self.file_list[idx]



transform = transforms.Compose([transforms.Resize((90,90)), transforms.ToTensor()])

#transforms.Resize((90,90))



human = dataset(train_files_human, train_dir+"humans/", mode = 'train',transform = transform)

n_human = dataset(train_files_non_human, train_dir+"non_human/", mode = 'train', transform = transform)

data = ConcatDataset([human, n_human])



train_len = int(0.8 * len(data))

validation_len = len(data) - train_len

train_set, validation_set = torch.utils.data.random_split(data, [train_len, validation_len])
dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
samples, label = iter(dataloader).next()

plt.figure(figsize=(16,34))

grid_imgs = torchvision.utils.make_grid(samples[:24])

np_grid_imgs = grid_imgs.numpy()

plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size = 3), nn.ReLU(), nn.MaxPool2d(2,2))

        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3), nn.ReLU(), nn.MaxPool2d(2,2))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = 3), nn.ReLU(), nn.MaxPool2d(2,2))

 #       self.layer4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size = 3), nn.ReLU(), nn.MaxPool2d(2,2))

 #       self.layer5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3), nn.ReLU(), nn.MaxPool2d(2,2))

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(9*9*128, 500)

        self.fc2 = nn.Linear(500,2)

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

  #      out = self.layer4(out)

  #      out = self.layer5(out)

        out = out.reshape(out.size(0), -1)

        out = self.drop_out(out)

        out = self.fc1(out)

        out = self.fc2(out)

        return out
model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
total_step = len(dataloader)

loss_list = []

acc_list = []



for epoch in range(10):

    for i, (images, labels) in enumerate(dataloader):

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss_list.append(loss.item()) 

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total = labels.size(0)

        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == labels).sum().item()

        acc_list.append(correct/total)

        if (i + 1) % 10 == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch+1, 20, i+1, total_step, loss.item(), (correct/total)*100))
validation_dataloader = DataLoader(validation_set, batch_size=32, shuffle=True)
model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in validation_dataloader:

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted==labels).sum().item()

    print("test accuracy:",(correct/total)*100)
test_dir = "/kaggle/input/jamp-hackathon-drive-1/test_set/"

test_set = dataset(test_files, test_dir, mode = 'train', transform = transform)

test_loader = DataLoader(test_set)
test_loader
solution = []

for image, name in test_loader:

    output = model(image)

    _, predicted = torch.max(output.data, 1)

    solution += predicted.tolist()
result = []

for i in range(len(test_files)):

    result.append([test_files[i].split(".")[0], solution[i]])