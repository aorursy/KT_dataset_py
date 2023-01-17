# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision

from torchvision import datasets, models, transforms

from torch.utils.data.sampler import SubsetRandomSampler



import syft as sy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# create workers, 

hook = sy.TorchHook(torch)



ada = sy.VirtualWorker(hook, 'ada')

bob = sy.VirtualWorker(hook, 'bob')

cyd = sy.VirtualWorker(hook, 'cyd')

secure_worker = sy.VirtualWorker(hook, 'secure_worker')
# define the transform

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, ), (0.5, ))

])



# load the datasets

#fulltrainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

#testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

fulltrainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)



train_size = int(len(fulltrainset)* 0.8)

valid_size = len(fulltrainset) - train_size



# split the dataset

trainset, validationset = torch.utils.data.random_split(fulltrainset, [train_size, valid_size])

trainset = trainset.dataset

validationset = validationset.dataset

federated_train_loader = sy.FederatedDataLoader(

    trainset.federate((ada,bob,cyd)), batch_size=64, shuffle=True)



federated_valid_loader = sy.FederatedDataLoader(

    validationset.federate((ada,bob,cyd)), batch_size=64, shuffle=True)



test_loader = torch.utils.data.DataLoader(

    testset, batch_size=64, shuffle=True)




# Lets check that our trainloader returns a pointer to a batch, and that transformations are applied

data, labels = next(iter(federated_train_loader))

data
print('ada {}, bob {}, cyd {}'.format(len(ada._objects), len(bob._objects), len(cyd._objects)))
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.fc1 = nn.Linear(784, 512)

        self.fc2 = nn.Linear(512, 10)



    def forward(self, x):

        x = x.view(-1, 784)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(federated_train_loader): # <-- now it is a distributed dataset

        # PySyft: send the model to the right location

        model.send(data.location) 

        

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        

        # PySyft: get the smarter model back

        model.get()

        

        if batch_idx % args.log_interval == 0:

            # Pysyft: get the loss back

            loss = loss.get()

            

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size, #batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))

            

    print('finished training')            
def test(args, model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss

            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 

            correct += pred.eq(target.view_as(pred)).sum().item()



    test_loss /= len(test_loader.dataset)



    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))
class Arguments():

    def __init__(self):

        self.batch_size = 64

        self.test_batch_size = 1000

        self.epochs = 10

        self.lr = 0.01

        self.momentum = 0.5

        self.no_cuda = False

        self.seed = 1

        self.log_interval = 20

        self.save_model = False



args = Arguments()



use_cuda = not args.no_cuda and torch.cuda.is_available()



torch.manual_seed(args.seed)



device = torch.device("cuda" if use_cuda else "cpu")



kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# try out a pretrained model

model = models.densenet161(pretrained=True)



for param in model.parameters():

    param.requires_grad = True

    

fc_in = model.classifier.in_features



transferclassifier = nn.Sequential(

                        nn.BatchNorm1d(fc_in),

                        nn.Linear(fc_in, 10)

                        )



#model.fc = transferclassifier # resnet

model.classifier = transferclassifier
#model = Model().to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr) # TODO momentum is not supported yet



for epoch in range(1, args.epochs + 1):

    train(args, model, device, federated_train_loader, optimizer, epoch)

    test(args, model, device, test_loader)



if (args.save_model):

    torch.save(model.state_dict(), "fashion_mnist_cnn.pt")
list(model.parameters())




print(f'objects of ada= {len(ada._objects)}, bob= {len(bob._objects)}, cyd= {len(cyd._objects)}')



ada.clear_objects()

bob.clear_objects()

cyd.clear_objects()