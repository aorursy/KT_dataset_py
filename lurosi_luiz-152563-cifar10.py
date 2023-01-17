import torch

import torchvision

import torchvision.transforms as transforms



transform = transforms.Compose(

    [transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

)



# ao usar o Kaggle, o root deve apontar para '../input/cifar10_pytorch/data'



trainset = torchvision.datasets.CIFAR10(root='../input/cifar10_pytorch/data', train=True,

                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='../input/cifar10_pytorch/data', train=False,

                                       download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,

                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import torch.nn as nn

import torch.nn.functional as F



class Net_1(nn.Module):

    def __init__(self):

        super(Net_1, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 12 * 12, 1504)

        self.fc2 = nn.Linear(1504, 1024)

        self.fc3 = nn.Linear(1024, 512)

        self.fc4 = nn.Linear(512, 84)

        self.fc5 = nn.Linear(84, 40)

        self.fc6 = nn.Linear(40, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 12 * 12)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = F.relu(self.fc5(x))

        x = self.fc6(x)

        return x

    

class Net_2(nn.Module):

    def __init__(self):

        super(Net_2, self).__init__()

        self.conv1 = nn.Conv2d(3, 16,1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 4 * 4, 102)

        self.fc2 = nn.Linear(102, 44)

        self.fc3 = nn.Linear(44, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x

    

class Net_3(nn.Module):

    def __init__(self):

        super(Net_3, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5)

        self.conv1_bn = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 24, 3)

        self.conv2_bn = nn.BatchNorm2d(24)

        

        self.fc1 = nn.Linear(24 * 6 * 6, 360)

        self.fc2 = nn.Linear(360, 104)

        self.fc3 = nn.Linear(104, 42)

        self.fc4 = nn.Linear(42, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv1_bn(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.conv2_bn(x)

        x = self.pool(x)

        x = x.view(-1, 24 * 6 * 6) 

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)



        return x
