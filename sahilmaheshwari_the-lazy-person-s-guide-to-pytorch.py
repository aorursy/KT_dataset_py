# Import the required libraries 

import pandas as pd

import numpy as np

import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim 

from torchvision.transforms import transforms

from torch.utils.data import DataLoader

from torch.utils.data import Dataset





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#reading the data

test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

# get the image pixel values and labels

train_labels = train.iloc[:, 0]

train_images = train.iloc[:, 1:]

test_images = test.iloc[:, 0:]


def get_device():

    if torch.cuda.is_available():

        device = 'cuda:0'

    else:

        device = 'cpu'

    return device

device = get_device()
transform = transforms.Compose(

    [transforms.ToPILImage(),

     transforms.ToTensor(),

     transforms.Normalize((0.5, ), (0.5, ))

])
class MNISTDataset(Dataset):

    def __init__(self, images, labels=None, transforms=None):

        self.X = images

        self.y = labels

        self.transforms = transforms

         

    def __len__(self):

        return (len(self.X))

    

    def __getitem__(self, i):

        data = self.X.iloc[i, :]

        data = np.asarray(data).astype(np.uint8).reshape(28, 28, 1)

        

        if self.transforms:

            data = self.transforms(data)

            

        if self.y is not None:

            return (data, self.y[i])

        else:

            return data

train_data = MNISTDataset(train_images, train_labels, transform)

test_data = MNISTDataset(test_images, transform)

# dataloaders

trainloader = DataLoader(train_data, batch_size=128, shuffle=True)

testloader = DataLoader(test_data, batch_size=128, shuffle=True)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, 

                               kernel_size=5, stride=1)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, 

                               kernel_size=5, stride=1)

        self.fc1 = nn.Linear(in_features=800, out_features=500)

        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

net = Net().to(device)

print(net)
#loss

criterion = nn.CrossEntropyLoss()

# optimizer

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
def train(net, trainloader):

    for epoch in range(4): # no. of epochs

        running_loss = 0

        for data in trainloader:

            # data pixels and labels to GPU if available

            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            # set the parameter gradients to zero

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            # propagate the loss backward

            loss.backward()

            # update the gradients

            optimizer.step()

 

            running_loss += loss.item()

        print('[Epoch %d] loss: %.3f' %

                      (epoch + 1, running_loss/len(trainloader)))

 

    print('Done Training')

x = torch.empty(0, 3)

def test(net, testloader):

    correct = 0

    total = 0

    with torch.no_grad():

        for data in testloader:

            inputs = data[0].to(device, non_blocking=True)

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            predictions=torch.cat((x,predicted),0)

#train(net, trainloader)

test(net, testloader)        

submission = pd.concat(

    [pd.Series(range(1,28001),name = "ImageId"),predictions],

    axis = 1

)

submission.to_csv("cnn_mnist_datagen.csv",index=False)