import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from torch import nn

from torch.utils import model_zoo

import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler




class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, 11, stride=4),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 192, 5, padding=2),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(192, 384, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),

        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(

            nn.Dropout(),

            nn.Linear(256 * 6 * 6, 4096),

            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(4096, 4096),

            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),

        )



    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
def alexnet_classifier(num_classes):

    classifier = nn.Sequential(

            nn.Dropout(),

            nn.Linear(256 * 6 * 6, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Linear(64, num_classes),

            nn.Softmax(),

        )

    return classifier



def alexnet(num_classes, pretrained=False, **kwargs):

    """AlexNet model architecture from the

    `"One weird trick..." `_ paper.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = AlexNet(**kwargs)

    if pretrained:

        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))

        for p in model.features.parameters():

            p.requires_grad=False

    classifier = alexnet_classifier(num_classes)

    model.classifier = classifier

    

    return model
torchvision.transforms.functional.resize

transform = transforms.Compose(

    [

     transforms.Resize(size=(224, 224)),

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
def show_weights(model, i=0):

    filter = model.features[i].weight.cpu().data

    filter = (1 / (2 * filter.max())) * filter + 0.5 #Normalizing the values to [0,1]

    print(filter.shape)

    img = torchvision.utils.make_grid(filter)

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

    imshow(img)
show_weights(alexnet(10, True))
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
net = alexnet(num_classes=10, pretrained=False)



use_cuda = True

if use_cuda and torch.cuda.is_available():

    print("using cuda")

    net.cuda()

learning_rate = 1e-3

optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

train(net, optimizer, trainloader, testloader, criterion,  n_epoch = 50,

      train_acc_period = 10, test_acc_period = 1000)

show_weights(net)

accuracy(net, testloader, cuda=use_cuda)
transnet = alexnet(num_classes=10, pretrained=True)



use_cuda = True

if use_cuda and torch.cuda.is_available():

    print("using cuda")

    transnet.cuda()

learning_rate = 1e-3

optimizer = torch.optim.Adam(transnet.parameters(),lr=learning_rate)

train(transnet, optimizer, trainloader, testloader, criterion,  n_epoch = 50,

      train_acc_period = 10, test_acc_period = 1000)

show_weights(transnet)

accuracy(transnet, testloader, cuda=use_cuda)