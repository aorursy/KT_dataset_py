import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from torch import nn

import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
class Inception(nn.Module):

    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):

        super(Inception, self).__init__()



        self.path1 = nn.Sequential(

            nn.Conv2d(in_planes, kernel_1_x, 1),

            nn.ReLU(inplace=True),

        )

        self.path2 = nn.Sequential(

            nn.Conv2d(in_planes, kernel_3_in, 1),

            nn.ReLU(inplace=True),

            nn.Conv2d(kernel_3_in, kernel_3_x, 3, padding=1),

            nn.ReLU(inplace=True),

        )

        self.path3 = nn.Sequential(

            nn.Conv2d(in_planes, kernel_5_in, 1),

            nn.ReLU(inplace=True),

            nn.Conv2d(kernel_5_in, kernel_5_x, 5, padding=2),

            nn.ReLU(inplace=True),

        )

        self.path4 = nn.Sequential(

            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),

            nn.Conv2d(in_planes, pool_planes, 1),

            nn.ReLU(inplace=True),

        )



    def forward(self, x):

        x1 = self.path1(x)

        x2 = self.path2(x)

        x3 = self.path3(x)

        x4 = self.path4(x)

        return torch.cat([x1, x2, x3, x4], 1)





class GoogLeNet(nn.Module):

    def __init__(self, input_dim=3):

        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(

            nn.Conv2d(input_dim, 192, kernel_size=3, padding=1),

            nn.BatchNorm2d(192),

            nn.ReLU(True),

        )

        

        self.layer1 = Inception(192,  64,  96, 128, 16, 32, 32)

        

        self.layer2 = Inception(256, 128, 128, 192, 32, 96, 64)

        

        self.layer3 = Inception(480, 192,  96, 208, 16,  48,  64)

        

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        

        self.avgpool = nn.AvgPool2d(8, stride=1)

        self.linear = nn.Linear(512, 10)

        



    def forward(self, x):

        x = self.pre_layers(x)



        x = self.layer1(x)

        x = self.max_pool(x)

        x = self.layer2(x)

        x = self.max_pool(x)

        x = self.layer3(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x
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
net = GoogLeNet()



use_cuda = True

if use_cuda and torch.cuda.is_available():

    print("using cuda")

    net.cuda()

learning_rate = 1e-3

optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

train(net, optimizer, trainloader, testloader, criterion,  n_epoch = 50,

      train_acc_period = 10, test_acc_period = 1000)

accuracy(net, testloader, cuda=use_cuda)