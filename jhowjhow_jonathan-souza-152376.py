import torch

import torchvision

import torchvision.transforms as transforms



transform = transforms.Compose(

    [transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

)



# ao usar o Kaggle, o root deve apontar para '../input/cifar10_pytorch/data'



trainset = torchvision.datasets.CIFAR10(root='../input/cifar10-pytorch/cifar10_pytorch/data', train=True,

                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='../input/cifar10-pytorch/cifar10_pytorch/data', train=False,

                                       download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,

                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)

        self.dropout = nn.Dropout(0.5)        

        self.fc2 = nn.Linear(in_features=500, out_features=50)

        self.fc3 = nn.Linear(in_features=50, out_features=2)

        

    def forward(self, X):

        X = F.relu(self.conv1(X))

        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv2(X))

        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv3_bn(self.conv3(X)))

        X = F.max_pool2d(X, 2)

        X = X.view(X.shape[0], -1)

        X = F.relu(self.fc1(X))

        X = self.dropout(X)

        X = F.relu(self.fc2(X))

        X = self.fc3(X)

        return X

    

    net = Net()
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv3 = nn.Conv2d(16, 26, 5)

        self.conv3_bn = nn.BatchNorm2d(26)

        self.fc1 = nn.Linear(26 * 1 * 1, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.conv3_bn(x)

        x = x.view(-1, 26 * 1 * 1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3)

        self.conv3 = nn.Conv2d(32, 64, 2)

        self.conv4 = nn.Conv2d(64, 96, 1)

        self.conv4_bn = nn.BatchNorm2d(96)

        self.fc1 = nn.Linear(96 * 2 * 2, 120)

        self.fc2 = nn.Linear(120, 84)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = F.relu(self.conv4(x))

        x = self.conv4_bn(x)

        #print(x.shape)

        x = x.view(-1, 96 * 2 * 2)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        return x





net = Net()
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 2000))

            running_loss = 0.0



print('Finished Training')
correct = 0

total = 0

with torch.no_grad():

    for data in testloader:

        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))