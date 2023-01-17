import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data preprocessing

import torch

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms



from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename);



# Any results you write to the current directory are saved as output.
labels = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")

submission = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv")



train_path = '/kaggle/input/aerial-cactus-identification/train/train/'

test_path = '/kaggle/input/aerial-cactus-identification/test/test/'
labels.head()
labels['has_cactus'].value_counts()
label = 'Has Cactus', 'Hasn\'t Cactus'

plt.figure(figsize = (8,8))

plt.pie(labels.groupby('has_cactus').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)

plt.show()
import matplotlib.image as img

fig,ax = plt.subplots(1,5,figsize = (15,3))



for i,idx in enumerate(labels[labels['has_cactus'] == 1]['id'][-5:]):

    path = os.path.join(train_path,idx)

    ax[i].imshow(img.imread(path))
fig,ax = plt.subplots(1,5,figsize = (15,3))

for i,idx in enumerate(labels[labels['has_cactus'] == 0]['id'][:5]):

    path = os.path.join(train_path,idx)

    ax[i].imshow(img.imread(path))
class CactusDataset(Dataset):

    def __init__(self, data, path , transform = None):

        super().__init__()

        self.data = data.values

        self.path = path

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self,index):

        img_name,label = self.data[index]

        img_path = os.path.join(self.path, img_name)

        image = img.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label

        

        
means = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



train_transform = transforms.Compose([transforms.ToPILImage(),

                                      transforms.ToTensor(),

                                      transforms.Normalize(means,std)])



test_transform = transforms.Compose([transforms.ToPILImage(),

                                     transforms.ToTensor(),

                                     transforms.Normalize(means,std)])



valid_transform = transforms.Compose([transforms.ToPILImage(),

                                     transforms.ToTensor(),

                                     transforms.Normalize(means,std)])

                        
train, valid_data = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)
train_data = CactusDataset(train, train_path, train_transform )

valid_data = CactusDataset(valid_data, train_path, valid_transform )

test_data = CactusDataset(submission, test_path, test_transform )


num_epochs = 50

num_classes = 2

batch_size = 64

learning_rate = 0.003

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)

valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)

test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)
epochs = 50

batch_size = 64

learning_rate = 0.003

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device
import torch.nn as nn

class CactusCNN(nn.Module):

    def __init__(self):

        super(CactusCNN,self).__init__()

        self.conv1 = nn.Sequential(

                     nn.Conv2d(3, 32, 2, 2, 0),

                     nn. BatchNorm2d(32),

                     nn.ReLU())

        # (32 - 2 + 0) / 2 + 1 = 16

        # 32 * 16 * 16

        self.conv2 = nn.Sequential(

                     nn.Conv2d(32,64,2,2,0),

                     nn.BatchNorm2d(64),

                     nn.ReLU())

        # (16 - 2 + 0) / 2 + 1 = 8

        # 64 * 8 * 8

        self.conv3 = nn.Sequential(

                     nn.Conv2d(64, 128, 2, 2,0),

                     nn.BatchNorm2d(128),

                     nn.ReLU())

        # (8 - 2 + 0) / 2 + 1 = 4

        # 128 * 4 * 4

        self.conv4 = nn.Sequential(nn.Conv2d(128,256,4,2,0),

                                   nn.BatchNorm2d(256),

                                   nn.ReLU())

        #(4 - 4 + 0) / 2 + 1 = 1

        # 256 * 1 * 1

        self.fc = nn.Sequential(nn.Linear(256 * 1 * 1, 512),

                                nn.ReLU(),

                                nn.Dropout(0.2),

                                nn.Linear(512, 128),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(128,2))

    def forward(self,x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        #flatten the image

        x = x.view(x.shape[0],-1)

        x = self.fc(x)

        return x

                    
model = CactusCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
# Training the mode

for epoch in range(epochs):

    for i,(images,labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device)

        

        

        pred = model(images)

        loss = criterion(pred, labels)

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print('Epoch: {}/{}, Loss: {}'.format(epoch+1, epochs, loss.item()))

        
model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in valid_loader:

        images = images.to(device)

        labels = labels.to(device)

        

        preds = model(images)

        _, predicted = torch.max(preds.data, 1)

        total += labels.size(0)

        correct += (labels == predicted).sum().item()

    print('Test Accuracy {} %'.format(100 * correct / total))
model.eval()

predict = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.to(device), target.to(device)

    output = model(data)



    pred = output[:,1].detach().cpu().numpy()

    for i in pred:

        predict.append(i)



submission['has_cactus'] = predict

submission.to_csv('submission.csv', index=False)