import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from torch import nn

import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
class ResNetBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):

        super(ResNetBlock, self).__init__()

        

        self.f = nn.Sequential(

            nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, 3, padding=1),

        )

        self.unity = nn.Sequential()

        if stride == 2:

            self.unity = nn.Conv2d(in_planes, planes, 1, stride=1)

        self.act = nn.ReLU(inplace=True)



    def forward(self, x):

        x = self.f(x)

        y = self.unity(x)

        x = x+y

        return self.act(x)





class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, num_filters=16, input_dim=3):

        super(ResNet, self).__init__()

        self.in_planes = num_filters



        self.conv1 = nn.Conv2d(input_dim, num_filters, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(num_filters)

        

        self.layer1 = self._make_layer(block, num_filters, num_blocks[0], 1)

        self.layer2 = self._make_layer(block, num_filters, num_blocks[1], 2)

        self.layer3 = self._make_layer(block, num_filters, num_blocks[2], 2)

        self.layer4 = self._make_layer(block, num_filters, num_blocks[3], 2)

        

        self.linear = nn.Linear(num_filters, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):

        layers = []

        layers.append(block(self.in_planes, planes, stride=stride))

        for i in range(1, num_blocks):

            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out





# (1 + 2*(1 + 1) + 2*(1 + 1) + 2*(1 + 1) + 2*(1 + 1)) + 1 = 18

def ResNet18():

    return ResNet(ResNetBlock, [2,2,2,2])
class SimpleBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):

        super(SimpleBlock, self).__init__()

        

        self.f = nn.Sequential(

            nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, 3, padding=1),

            nn.ReLU(inplace=True),

        )



    def forward(self, x):

        return self.f(x)
torchvision.transforms.functional.resize

transform = transforms.Compose(

    [

     transforms.Resize(size=(32, 32)),

     transforms.ToTensor(),

     transforms.Normalize((0.5,), (0.5,)),

])

     



batch_size = 64



idx_train = np.arange(50000)

np.random.shuffle(idx_train)

idx_train = idx_train[:1000]



trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=2,

                                         sampler=SubsetRandomSampler(idx_train))



idx_test = np.arange(10000)

np.random.shuffle(idx_test)

idx_test = idx_train[:1000]



testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

testloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=2)





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
criterion = nn.CrossEntropyLoss()



def accuracy(net, test_loader, cuda=True):

    net.eval()

    correct = 0

    total = 0

    loss = 0

    with torch.no_grad():

        for data in test_loader:

            images, labels = data

            if cuda:

                images = images.type(torch.cuda.FloatTensor)

                labels = labels.type(torch.cuda.LongTensor)

            outputs = net(images)

            # loss+= criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            # if total > 100:

                # break

    net.train()

    print('Accuracy of the network on the test images: %d %%' % (

        100 * correct / total))

    # return (100.0 * correct / total, loss/total)

    return 100.0 * correct / total



def train(net, optimizer, train_loader, test_loader, loss,  n_epoch = 5,

          train_acc_period = 100, test_acc_period = 5, cuda=True):

    loss_train = []

    loss_test = []

    total = 0

    for epoch in range(n_epoch):  # loop over the dataset multiple times

        running_loss = 0.0

        running_acc = 0.0

        for i, data in enumerate(train_loader, 0):

            # get the inputs

            inputs, labels = data

            if cuda:

                inputs = inputs.type(torch.cuda.FloatTensor)

                labels = labels.type(torch.cuda.LongTensor)

            # print(inputs.shape)

            # zero the parameter gradients

            optimizer.zero_grad()



            # forward + backward + optimize

            outputs = net(inputs)

          

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total += labels.size(0)

            # print statistics

            running_loss = 0.33*loss.item()/labels.size(0) + 0.66*running_loss

            _, predicted = torch.max(outputs.data, 1)

            correct = (predicted == labels).sum().item()/labels.size(0)

            running_acc = 0.3*correct + 0.66*running_acc

            if i % train_acc_period == train_acc_period-1:

                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))

                print('[%d, %5d] acc: %.3f' %(epoch + 1, i + 1, running_acc))

                running_loss = 0.0

                total = 0

                # break

        if epoch % test_acc_period == test_acc_period-1:

            cur_acc, cur_loss = accuracy(net, test_loader, cuda=cuda)

            print('[%d] loss: %.3f' %(epoch + 1, cur_loss))

            print('[%d] acc: %.3f' %(epoch + 1, cur_acc))

      

    print('Finished Training')
net = ResNet(SimpleBlock, [2,2,2,2])



use_cuda = True

if use_cuda and torch.cuda.is_available():

    print("using cuda")

    net.cuda()

learning_rate = 1e-3

optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

train(net, optimizer, trainloader, testloader, criterion,  n_epoch = 50,

      train_acc_period = 10, test_acc_period = 1000)

accuracy(net, testloader, cuda=use_cuda)
net = ResNet18()



use_cuda = True

if use_cuda and torch.cuda.is_available():

    print("using cuda")

    net.cuda()

learning_rate = 1e-3

optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

train(net, optimizer, trainloader, testloader, criterion,  n_epoch = 50,

      train_acc_period = 10, test_acc_period = 1000)

accuracy(net, testloader, cuda=use_cuda)