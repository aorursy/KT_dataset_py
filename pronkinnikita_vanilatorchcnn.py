import numpy as np 
import pandas as pd

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
train = pd.read_csv('../input/train.csv').values
test = pd.read_csv('../input/test.csv').values
''' doing inheritance from class torch.unils.data.Dataset,
    look https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
class MNIST(Dataset):

    def __init__(self, frame, train = True, transform=None):
        
        self.frame = frame
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if self.train:
            image = self.frame[idx, 1: ].reshape((28, 28, 1)).astype(np.float32)
            label = self.frame[idx, 0]
            sample = {'image': image, 'label': label}
            if self.transform:
                sample['image'] = self.transform(sample['image'])
        else:
            image = self.frame[idx].reshape((28, 28, 1)).astype(np.float32)
            sample = image
            if self.transform:
                sample = self.transform(sample)
        
        return sample
mnist = MNIST(frame=train, train=True, transform=transforms.Compose([transforms.ToTensor()]))

dataloader = DataLoader(mnist, batch_size=8,
                        shuffle=True, num_workers=4)
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
class VanilaCNN(nn.Module):
    def __init__(self):
        super(VanilaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x
cnn = VanilaCNN()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
import warnings
warnings.filterwarnings('ignore')
for epoch in range(2):
    
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['image'], data['label']
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
cnn = VanilaCNN()
cnn = cnn.to('cuda:0')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['image'].to('cuda'), data['label'].to('cuda')
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
mnist_test = MNIST(frame=test, train=False, transform=transforms.Compose([transforms.ToTensor()]))

dataloader_test = DataLoader(mnist_test, batch_size=8,
                        shuffle=False, num_workers=4)
res = []
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        inputs = data.to('cuda')
        outputs = torch.exp(cnn(inputs)).argmax(dim=1).tolist()
        res += outputs
    print('Finished Training')   
len(res)
res = np.array(res)
res = pd.DataFrame(res, columns=['Label'])
res['ImageId'] = np.arange(1, len(res) + 1)
res = res[['ImageId', 'Label']]
res.head()
example_sub = pd.read_csv('../input/sample_submission.csv')
res.to_csv('res.csv', index=False)
res2 = pd.read_csv('res.csv',)
res2.head()
example_sub.info()
example_sub.head()
!ls
import sys

plt.imshow(test[7].reshape((28, 28)))
plt.imshow(data[7].numpy()[0, : , :])
outputs.argmax(dim=1)
