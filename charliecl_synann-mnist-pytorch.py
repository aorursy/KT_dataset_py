# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# !pip install easydict
#

#

# SynaNN for Image Classification with MNIST Dataset in Pytorch

#

# Copyright (c) 2019, Chang LI. All rights reserved.

#

# Open source, MIT License.

#

#



# header

from __future__ import print_function



import math

import argparse



import torch

from torch.nn.parameter import Parameter

from torch.nn import init

from torch.nn import Module



import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

from torchvision import datasets, transforms



import matplotlib.pyplot as plt



train_losses = []

train_counter = []

test_counter = []

test_losses = []



class Synapse(nn.Module):

    r"""Applies a synapse function to the incoming data.`



    Args:

        in_features:  size of each input sample

        out_features: size of each output sample

        bias:         if set to ``False``, the layer will not learn an additive bias.

                      Default: ``True``



    Shape:

        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of

             additional dimensions and :math:`H_{in} = \text{in\_features}`

        - Output: :math:`(N, *, H_{out})` where all but the last dimension

             are the same shape as the input and :math:`H_{out} = \text{out\_features}`.



    Attributes:

        weight: the learnable weights of the module of shape

            	:math:`(\text{out\_features}, \text{in\_features})`. The values are

            	initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where

            	:math:`k = \frac{1}{\text{in\_features}}`

        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.

                If :attr:`bias` is ``True``, the values are initialized from

                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where

                :math:`k = \frac{1}{\text{in\_features}}`



    Examples::



        >>> m = Synapse(64, 64)

        >>> input = torch.randn(128, 20)

        >>> output = m(input)

        >>> print(output.size())

        torch.Size([128, 30])

    """

    __constants__ = ['bias', 'in_features', 'out_features']



    def __init__(self, in_features, out_features, bias=True):

        super(Synapse, self).__init__()

        self.in_features = in_features

        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias:

            self.bias = Parameter(torch.Tensor(out_features))

        else:

            self.register_parameter('bias', None)

        self.reset_parameters()



    def reset_parameters(self):

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)

            bound = 1 / math.sqrt(fan_in)

            init.uniform_(self.bias, -bound, bound)

    

    # synapse core

    def forward(self, input):

        # shapex = matrix_diag(input)

        diagx = torch.stack(tuple(t.diag() for t in torch.unbind(input,0)))

        shapex = diagx.view(-1, self.out_features)

        betax = torch.log1p(-shapex @ self.weight.t())

        row = betax.size()

        allone = torch.ones(int(row[0]/self.out_features), row[0])

        if torch.cuda.is_available():

          allone = allone.cuda()

        return torch.exp(torch.log(input) + allone @ betax) # + self.bias)    



    def extra_repr(self):

        return 'in_features={}, out_features={}, bias={}'.format(

            self.in_features, self.out_features, self.bias is not None

        )



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        

        # fully connected with synapse function

        self.fc1 = nn.Linear(320, 64)

        self.fcn = Synapse(64,64)

        self.fcb = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 10)



    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = F.softmax(x, dim=1)

        

        # fcn is the output of synapse

        x = self.fcn(x)

        # fcb is the batch no)rmal 

        x = self.fcb(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)



def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))

            train_losses.append(loss.item())

            train_counter.append(

                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

            torch.save(model.state_dict(), 'model.pth')

            torch.save(optimizer.state_dict(), 'optimizer.pth')



def test(args, model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0



    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

  

def main():

    print(torch.version.__version__)

    

    # Training settings

    import easydict

    args = easydict.EasyDict({

      "batch_size": 100,

      "test_batch_size": 100,

      "epochs": 10,

      "lr": 0.012,

      "momentum": 0.5,

      "no_cuda": False,

      "seed": 5,

      "log_interval":100

    })

    

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    torch.backends.cudnn.enabled = False

   

    device = torch.device("cuda:0" if use_cuda else "cpu")

    

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(

        datasets.MNIST('../data', train=True, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize((0.1307,), (0.3081,))

                       ])),

        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(

        datasets.MNIST('../data', train=False, transform=transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize((0.1307,), (0.3081,))

                       ])),

        batch_size=args.test_batch_size, shuffle=True, **kwargs)



    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

                          

    test_counter = [i*len(train_loader.dataset) for i in range(args.epochs)]

    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)

        test(args, model, device, test_loader)

    

    # draw curves

    fig = plt.figure()

    plt.plot(train_counter, train_losses, color='blue')

    plt.scatter(test_counter, test_losses, color='red')

    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')

    plt.xlabel('number of training examples seen')

    plt.ylabel('negative log likelihood loss')

    fig

        

if __name__ == '__main__':

  main()