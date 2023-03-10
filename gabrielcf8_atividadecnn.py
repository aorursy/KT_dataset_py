import torch

import torchvision

import torchvision.transforms as transforms



transform = transforms.Compose(

    [transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

)



# ao usar o Kaggle, o root deve apontar para '../input/cifar10-pytorch/cifar10_pytorch/data'



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
import matplotlib.pyplot as plt

import numpy as np



# functions to show an image

def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()





# get some random training images

dataiter = iter(trainloader)

images, labels = dataiter.next()



# show images

imshow(torchvision.utils.make_grid(images))



# print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()

'''
'''import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)'''
'''for epoch in range(2):  # loop over the dataset multiple times



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



print('Finished Training')'''
#MODEL NUMBER ONE

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



class Net1(nn.Module):

    def __init__(self):

        super(Net1, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, padding=1)

        self.conv3 = nn.Conv2d(16, 32, 5, padding=1)

        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 2 * 2, 128)

        self.fc2 = nn.Linear(128, 100)

        self.fc3 = nn.Linear(100, 50)

        self.fc4 = nn.Linear(50, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = self.conv3_bn(x)

        #print(x.shape)

        x = x.view(-1, 32 * 2 * 2)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net1 = Net1()

net1.to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net1(inputs)

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

print('----------------------------------')

correct = 0

total = 0



with torch.no_grad():

    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net1(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network 1 on the 10000 test images: %d %%' % (

    100 * correct / total))
#MODEL NUMBER TWO

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



class Net2(nn.Module):

    def __init__(self):

        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=1)

        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 2 * 2, 256)

        self.fc2 = nn.Linear(256, 200)

        self.fc3 = nn.Linear(200, 100)

        self.fc4 = nn.Linear(100, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = self.conv3_bn(x)

        #print(x.shape)

        x = x.view(-1, 64 * 2 * 2)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net2 = Net2()

net2.to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net2(inputs)

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

print('----------------------------------')

correct = 0

total = 0



with torch.no_grad():

    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net2(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network 2 on the 10000 test images: %d %%' % (

    100 * correct / total))
#MODEL NUMBER THREE

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



class Net3(nn.Module):

    def __init__(self):

        super(Net3, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)

        self.conv2_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 6 * 6, 2034)

        self.fc2 = nn.Linear(2034, 1500)

        self.fc3 = nn.Linear(1500, 1000)

        self.fc4 = nn.Linear(1000, 500)

        self.fc5= nn.Linear(500, 50)

        self.fc6 = nn.Linear(50, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        #print(x.shape)

        x = x.view(-1, 64 * 6 * 6)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = F.relu(self.fc5(x))

        x = self.fc6(x)

        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net3 = Net3()

net3.to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net3.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net3(inputs)

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

print('----------------------------------')

correct = 0

total = 0



with torch.no_grad():

    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net3(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network 3 on the 10000 test images: %d %%' % (

    100 * correct / total))
'''dataiter = iter(testloader)

images, labels = dataiter.next()



# print images

imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))'''
'''outputs = net(images)'''
'''_, predicted = torch.max(outputs, 1)



print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]

                              for j in range(4)))'''
'''correct = 0

total = 0

with torch.no_grad():

    for data in testloader:

        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))'''
'''class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))

with torch.no_grad():

    for data in testloader:

        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(4):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(10):

    print('Accuracy of %5s : %2d %%' % (

        classes[i], 100 * class_correct[i] / class_total[i]))'''
'''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



net = Net()

net.to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to(device), data[1].to(device)



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



print('Finished Training')'''
'''correct = 0

total = 0



with torch.no_grad():

    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))'''
'''print(net)'''
'''import numpy as np



# precisa colocar os prints na defini????o da rede

x = torch.Tensor(np.ones((1,3,32,32)))

net.forward(x)'''