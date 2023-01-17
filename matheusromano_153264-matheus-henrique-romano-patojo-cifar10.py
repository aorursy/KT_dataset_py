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

        self.fc4 = nn.Linear(84,50)

        self.fc5 = nn.Linear(50,30)

        self.fc6 = nn.Linear(30,15)

        self.fc3 = nn.Linear(15, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc4(x))

        x = F.relu(self.fc5(x))

        x = F.relu(self.fc6(x))

        x = self.fc3(x)

        return x





net = Net()
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv3 = nn.Conv2d(16,16,2)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 3 * 3, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        

        x = F.relu(self.conv3(x))

        

        x = F.relu(self.conv3(x))

        x = x.view(-1, 16 * 3 * 3)

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

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84,70)

        self.fc4 = nn.Linear(70,60)

        self.fc5 = nn.Linear(60,50)

        self.fc6 = nn.Linear(50,40)

        self.fc7 = nn.Linear(40,30)

        self.fc8 = nn.Linear(30,15)

        self.fc9 = nn.Linear(15, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = F.relu(self.fc5(x))

        x = F.relu(self.fc6(x))

        x = F.relu(self.fc7(x))

        x = F.relu(self.fc8(x))

        x = self.fc9(x)

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
dataiter = iter(testloader)

images, labels = dataiter.next()



# print images

imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)



print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]

                              for j in range(4)))
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
class_correct = list(0. for i in range(10))

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

        classes[i], 100 * class_correct[i] / class_total[i]))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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



print('Finished Training')
correct = 0

total = 0



with torch.no_grad():

    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))
print(net)
import numpy as np



# precisa colocar os prints na definição da rede

x = torch.Tensor(np.ones((1,3,32,32)))

net.forward(x)