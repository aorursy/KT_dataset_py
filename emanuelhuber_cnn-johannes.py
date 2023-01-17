import torch

import torchvision

import torchvision.transforms as transforms



transform = transforms.Compose(

    [transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

)



# ao usar o Kaggle, o root deve apontar para '../input/cifar10_pytorch/data'



trainset = torchvision.datasets.CIFAR10(root='./input/cifar10_pytorch/cifar10_pytorch/data', train=True,

                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='./input/cifar10_pytorch/cifar10_pytorch/data', train=False,

                                       download=True, transform=transform)

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
class Net2(nn.Module):

    def __init__(self):

        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(3, 18, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(18, 16, 5)

        self.conv3 = nn.Conv2d(16, 32, 3)

        self.conv2_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 32)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
# ZFNet, ficou pior :(



class Net3(nn.Module):

    def __init__(self):

        super(Net3, self).__init__()

        self.channels = 3

        self.conv1 = nn.Conv2d(self.channels, 96, kernel_size=7, stride=2, padding=1)

        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)

        nn.init.constant_(self.conv1.bias, 0.0)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.norm1 = nn.LocalResponseNorm(5)

        

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)

        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.02)

        nn.init.constant_(self.conv2.bias, 0.0)

        self.norm2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.02)

        nn.init.constant_(self.conv3.bias, 0.0)

        

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        nn.init.normal_(self.conv4.weight, mean=0.0, std=0.02)

        nn.init.constant_(self.conv4.bias, 0.0)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        

        self.ln1 = nn.Linear(256, 128)

        nn.init.normal_(self.ln1.weight, mean=0.0, std=0.02)

        nn.init.constant_(self.ln1.bias, 0.0)

        self.drop = nn.Dropout()

        

        self.ln2 = nn.Linear(128, 10)

        nn.init.normal_(self.ln2.weight, mean=0.0, std=0.02)

        nn.init.constant_(self.ln2.bias, 0.0)

        



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool1(x)

        x = self.norm1(x)

        x = F.relu(self.conv2(x))

        x = self.pool1(x)

        x = self.norm2(x)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = self.pool1(x)

        x = x.view(x.size(0), -1)

        x = self.ln1(x)

        x = self.drop(x)

        x = self.ln2(x)

        x = self.drop(x)

        return x
class Net4(nn.Module):

    def __init__(self):

        super(Net4, self).__init__()

        self.conv1 = nn.Conv2d(3, 18, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc3 = nn.Linear(18 * 14 * 14, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = x.view(-1, 18 * 14 * 14)

        x = self.fc3(x)

        return x
def calcAcc():

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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# nets = {"Net2": Net2, "Net3": Net3, "Net4": Net4}

# repetitions = 5

# res = {"Net2": [], "Net3": [], "Net4": []}



# for rep in range(repetitions):

#     for k, net in nets.items():

#         print("Training net %s at rep %d" % (k, rep))

#         net = net() 



#         # net = Net4()

#         net.to(device)



#         criterion = nn.CrossEntropyLoss().to(device)

#         optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



#         for epoch in range(2):  # loop over the dataset multiple times



#             running_loss = 0.0

#             for i, data in enumerate(trainloader, 0):

#                 # get the inputs; data is a list of [inputs, labels]

#                 inputs, labels = data[0].to(device), data[1].to(device)



#                 # zero the parameter gradients

#                 optimizer.zero_grad()



#                 # forward + backward + optimize

#                 outputs = net(inputs)

#                 loss = criterion(outputs, labels)

#                 loss.backward()

#                 optimizer.step()



#                 # print statistics

#                 running_loss += loss.item()

#                 if i % 2000 == 1999:    # print every 2000 mini-batches

#                     print('[%d, %5d] loss: %.3f' %

#                           (epoch + 1, i + 1, running_loss / 2000))

#                     running_loss = 0.0



#         correct = 0

#         total = 0



#         with torch.no_grad():

#             for data in testloader:

#                 images, labels = data[0].to(device), data[1].to(device)

#                 outputs = net(images)

#                 _, predicted = torch.max(outputs.data, 1)

#                 total += labels.size(0)

#                 correct += (predicted == labels).sum().item()



#         print('Accuracy of the network on the 10000 test images: %d %%' % (

#             100 * correct / total))



#         res[k].append( 100 * correct / total )



# # print('Finished Training')
# accs = [np.mean(res['Net2']), np.mean(res['Net3']), np.mean(res['Net4'])]

# print('Net2 mean acc: %lf' % accs[0])

# print('Net3 mean acc: %lf' % accs[1])

# print('Net4 mean acc: %lf' % accs[2])

# print('The best network is: %s' % list(res.keys())[np.argmax(accs)])
res