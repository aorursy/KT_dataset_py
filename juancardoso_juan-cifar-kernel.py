import torch

import torchvision

import torchvision.transforms as transforms



transform = transforms.Compose(

    [transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

)



# ao usar o Kaggle, o root deve apontar para '../input/cifar10_pytorch/data'



path_root = "../input/cifar10_pytorch/data"



trainset = torchvision.datasets.CIFAR10(root=path_root, train=True,

                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root=path_root, train=False,

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

        print(x.shape)

        x = self.pool(x)

        print(x.shape)

        x = F.relu(self.conv2(x))

        print(x.shape)

        x = self.pool(x)

        print(x.shape)

        x = self.conv2_bn(x)

        print(x.shape)

        x = x.view(-1, 16 * 5 * 5)

        print(x.shape)

        x = F.relu(self.fc1(x))

        print(x.shape)

        x = F.relu(self.fc2(x))

        print(x.shape)

        x = self.fc3(x)

        print(x.shape)

        

        return x





net = Net()
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.LastLayerSize = 96*2*2 

        

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 12, 5)

        self.conv3 = nn.Conv2d(12, 24, 5)

        self.conv4 = nn.Conv2d(24, 48, 5)

        self.conv5 = nn.Conv2d(48, 96, 5)

        self.conv2_bn = nn.BatchNorm2d(96)

        self.fc1 = nn.Linear(self.LastLayerSize, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

#         print("1 - {}".format(x.shape))

        

        x = F.relu(self.conv1(x))

#         print("2 - {}".format(x.shape))

        

#         x = self.pool(x)

#         print("3 - {}".format(x.shape))

        

        x = F.relu(self.conv2(x))

#         print("4 - {}".format(x.shape))

        

#         x = self.pool(x)

#         print("5 - {}".format(x.shape))

        

        x = F.relu(self.conv3(x))

#         print("6 - {}".format(x.shape))

        

#         x = self.pool(x)

#         print("7 - {}".format(x.shape))

        

        x = F.relu(self.conv4(x))

#         print("8 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("9 - {}".format(x.shape))

        

        x = F.relu(self.conv5(x))

#         print("10 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("11 - {}".format(x.shape))

        

        x = self.conv2_bn(x)

#         print("12 - {}".format(x.shape))

        

        x = x.view(-1, self.LastLayerSize)

#         print("13 - {}".format(x.shape))

        

        x = F.relu(self.fc1(x))

#         print("14 - {}".format(x.shape))

        

        x = F.relu(self.fc2(x))

#         print("15 - {}".format(x.shape))

        

        x = self.fc3(x)

#         print("16 - {}".format(x.shape))

        

        return x





net = Net()
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.LastLayerSize = 96*7*7 

        

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 1)

        self.conv2 = nn.Conv2d(6, 12, 5)

        self.conv3 = nn.Conv2d(12, 24, 5)

        self.conv4 = nn.Conv2d(24, 48, 5)

        self.conv5 = nn.Conv2d(48, 96, 5)

        self.conv2_bn = nn.BatchNorm2d(96)

        self.fc1 = nn.Linear(self.LastLayerSize, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

#         print("1 - {}".format(x.shape))

        

        x = F.relu(self.conv1(x))

#         print("2 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("3 - {}".format(x.shape))

        

        x = F.relu(self.conv2(x))

#         print("4 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("5 - {}".format(x.shape))

        

        x = F.relu(self.conv3(x))

#         print("6 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("7 - {}".format(x.shape))

        

        x = F.relu(self.conv4(x))

#         print("8 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("9 - {}".format(x.shape))

        

        x = F.relu(self.conv5(x))

#         print("10 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("11 - {}".format(x.shape))

        

        x = self.conv2_bn(x)

#         print("12 - {}".format(x.shape))

        

        x = x.view(-1, self.LastLayerSize)

#         print("13 - {}".format(x.shape))

        

        x = F.relu(self.fc1(x))

#         print("14 - {}".format(x.shape))

        

        x = F.relu(self.fc2(x))

#         print("15 - {}".format(x.shape))

        

        x = self.fc3(x)

#         print("16 - {}".format(x.shape))

        

        return x





net = Net()
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.LastLayerSize = 512*1*1 

        

        self.conv3_64 = nn.Conv2d(3, 64,  kernel_size=3, padding=1)

        self.conv64_64 = nn.Conv2d(64, 64,  kernel_size=3, padding=1)

        self.conv64_128 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)

        self.conv128_128 = nn.Conv2d(128, 128,  kernel_size=3, padding=1)

        self.conv128_256 = nn.Conv2d(128, 256,  kernel_size=3, padding=1)

        self.conv256_256 = nn.Conv2d(256, 256,  kernel_size=3, padding=1)

        self.conv256_512 = nn.Conv2d(256, 512,  kernel_size=3, padding=1)

        self.conv512_512 = nn.Conv2d(512, 512,  kernel_size=3, padding=1)

        

        self.conv_bn_3 = nn.BatchNorm2d(3)

        self.conv_bn_64 = nn.BatchNorm2d(64)

        self.conv_bn_128 = nn.BatchNorm2d(128)

        self.conv_bn_256 = nn.BatchNorm2d(256)

        self.conv_bn_512 = nn.BatchNorm2d(512)

        

        self.pool = nn.MaxPool2d(2, 2)

        

        self.fc1 = nn.Linear(self.LastLayerSize, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

#         print("1 - {}".format(x.shape))

    

        x = self.conv3_64(x)

        x = self.conv_bn_64(x)

        x = F.relu(x)

#         print("2 - {}".format(x.shape))

        

        x = self.conv64_64(x)

        x = self.conv_bn_64(x)

        x = F.relu(x)

#         print("3 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("4 - {}".format(x.shape))

        

        x = self.conv64_128(x)

        x = self.conv_bn_128(x)

        x = F.relu(x)

#         print("5 - {}".format(x.shape))



        x = self.conv128_128(x)

        x = self.conv_bn_128(x)

        x = F.relu(x)

#         print("6 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("7 - {}".format(x.shape))

        

        x = self.conv128_256(x)

        x = self.conv_bn_256(x)

        x = F.relu(x)

#         print("8 - {}".format(x.shape))

        

        x = self.conv256_256(x)

        x = self.conv_bn_256(x)

        x = F.relu(x)

#         print("9 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("10 - {}".format(x.shape))

        

        x = self.conv256_512(x)

        x = self.conv_bn_512(x)

        x = F.relu(x)

#         print("11 - {}".format(x.shape))

        

        x = self.conv512_512(x)

        x = self.conv_bn_512(x)

        x = F.relu(x)

#         print("12 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("13 - {}".format(x.shape))

        

        x = self.conv512_512(x)

        x = self.conv_bn_512(x)

        x = F.relu(x)

#         print("14 - {}".format(x.shape))

        

        x = self.conv512_512(x)

        x = self.conv_bn_512(x)

        x = F.relu(x)

#         print("15 - {}".format(x.shape))

        

        x = self.pool(x)

#         print("16 - {}".format(x.shape))

        

        x = x.view(-1, self.LastLayerSize)

#         print("17 - {}".format(x.shape))

        

        x = F.relu(self.fc1(x))

#         print("18 - {}".format(x.shape))

        

        x = F.relu(self.fc2(x))

#         print("19 - {}".format(x.shape))

        

        x = self.fc3(x)

#         print("20 - {}".format(x.shape))

        

        return x





net = Net()
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# for epoch in range(2):  # loop over the dataset multiple times



#     running_loss = 0.0

#     for i, data in enumerate(trainloader, 0):

#         # get the inputs; data is a list of [inputs, labels]

#         inputs, labels = data



#         # zero the parameter gradients

#         optimizer.zero_grad()



#         # forward + backward + optimize

#         outputs = net(inputs)

#         loss = criterion(outputs, labels)

#         loss.backward()

#         optimizer.step()



#         # print statistics

#         running_loss += loss.item()

#         if i % 2000 == 1999:    # print every 2000 mini-batches

#             print('[%d, %5d] loss: %.3f' %

#                   (epoch + 1, i + 1, running_loss / 2000))

#             running_loss = 0.0



# print('Finished Training')
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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# net = Net()

# net.to(device)



# criterion = nn.CrossEntropyLoss().to(device)

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



# for epoch in range(2):  # loop over the dataset multiple times



#     running_loss = 0.0

#     for i, data in enumerate(trainloader, 0):

#         # get the inputs; data is a list of [inputs, labels]

#         inputs, labels = data[0].to(device), data[1].to(device)



#         # zero the parameter gradients

#         optimizer.zero_grad()



#         # forward + backward + optimize

#         outputs = net(inputs)

#         loss = criterion(outputs, labels)

#         loss.backward()

#         optimizer.step()



#         # print statistics

#         running_loss += loss.item()

#         if i % 2000 == 1999:    # print every 2000 mini-batches

#             print('[%d, %5d] loss: %.3f' %

#                   (epoch + 1, i + 1, running_loss / 2000))

#             running_loss = 0.0



# print('Finished Training')
# correct = 0

# total = 0



# with torch.no_grad():

#     for data in testloader:

#         images, labels = data[0].to(device), data[1].to(device)

#         outputs = net(images)

#         _, predicted = torch.max(outputs.data, 1)

#         total += labels.size(0)

#         correct += (predicted == labels).sum().item()



# print('Accuracy of the network on the 10000 test images: %d %%' % (

#     100 * correct / total))
print(net)
import numpy as np



# precisa colocar os prints na definição da rede

x = torch.Tensor(np.ones((1,3,32,32)))

net.forward(x)