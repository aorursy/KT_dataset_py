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

from torch.utils import data

import cv2

from matplotlib.pyplot import imshow

import torchvision

import torch

import torch.nn as nn

import torch.nn.functional as F



trainPath = '../input/train/train/'

testPath = '../input/test/test/'



df = pd.read_csv('../input/train.csv')

print(df['has_cactus'].value_counts())

print(df.head())
!cd ../input/

!ls ../input/test/test
batch_size = 128

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Number of Train images: {}".format(len(os.listdir(trainPath))))

print("Number of Test images: {}".format(len(os.listdir(testPath))))
from PIL import Image



class Dataset(data.Dataset):

  'Characterizes a dataset for PyTorch'

  def __init__(self, data, dataPath, transform = None):

        'Initialization'

        self.data = data

        self.transform = transform

        self.dataPath = dataPath



  def __len__(self):

        'Denotes the total number of samples'

        return len(self.data)



  def __getitem__(self, index):

        'Generates one sample of data'

        # Select sample

        imgName = self.data.values[index][0]

        y = self.data.values[index][1]  

    

        # Load data and get label

        X = cv2.imread(os.path.join(self.dataPath,imgName))

        X = cv2.resize(X,(32,32))

#         X = cv2.cvtColor(X,cv2.COLOR_BGR2RGB)

#         X = Image.fromarray(X)

        if self.transform:

            X = self.transform(X)

        

        return X, y

    

    



from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms

from torch.utils.data import DataLoader



X_train, X_val = train_test_split(df, stratify=df.has_cactus, test_size=0.2)

image_transform = transforms.Compose([transforms.ToPILImage(),transforms.Pad(32, padding_mode='reflect'),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



train_dataset = Dataset(data = X_train, dataPath = trainPath, transform = image_transform)

test_dataset = Dataset(data = X_val, dataPath = trainPath, transform = image_transform)



loader_train = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

loader_valid = DataLoader(dataset = test_dataset, batch_size=batch_size//2, shuffle=False, num_workers=0)

class leNet(nn.Module):

    def __init__(self):

        super(leNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)

        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(128*13*13, 2)

        self.dropout = nn.Dropout(0.25)

    

    def forward(self,x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.dropout(x)

        

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.dropout(x)

        

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.dropout(x)

        x = x.view(-1,128*13*13)

        x = self.fc(x)

        return x



     

        

        

        
net = leNet().to(device)

print(net)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(net.parameters(), lr=0.001)

EPOCHS = 10

running_loss = 0

train_losses, test_losses = [], []



steps = len(loader_train)

print("Total steps = ",steps)

for epoch in range(EPOCHS):

    for n, (image, label) in enumerate(loader_train):

        image = image.to(device)

        label = label.to(device)

        

        outputs = net(image)

        loss = criterion(outputs, label)

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    print(f"Epoch: {epoch+1}/{EPOCHS}... Train loss: {loss.item():.3f}...")

    if (epoch+1) % 5 == 0:

        test_loss = 0

        net.eval()

        accuracy = 0

        with torch.no_grad():

            for i, (imageTest, labelTest) in enumerate(loader_valid):

                imageTest = imageTest.to(device)

                labelTest = labelTest.to(device)

                predictions = net.forward(imageTest)

                batch_loss = criterion(predictions, labelTest)

                test_loss += batch_loss.item()

                ps = torch.exp(predictions)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labelTest.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



            train_losses.append(running_loss/len(loader_train))

            test_losses.append(test_loss/len(loader_valid))                    

            print(f"Test loss: {test_loss/len(loader_valid):.3f}... Test accuracy: {accuracy/len(loader_valid):.3f}...")

            running_loss = 0

            net.train()

#torch.save(net.state_dict(), 'model.ckpt')
!ls
sub = pd.read_csv('../input/sample_submission.csv')

print(sub.head())

submission_dataset = Dataset(data = sub, dataPath = testPath, transform=image_transform)

loader_sub = DataLoader(dataset = submission_dataset, batch_size=32, shuffle=False, num_workers=0)
net.eval()



preds = []

for batch, (imageSub, labelSub) in enumerate(loader_sub):

    print(f"{batch+1}/{len(loader_sub)}")

    imageSub, labelSub = imageSub.cuda(), labelSub.cuda()

    output = net(imageSub)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)



sub['has_cactus'] = preds

sub.to_csv('sample_submission.csv', index=False)

sub.head()
!ls

!cd testPath