from __future__ import print_function

import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F



class Custom_Network(nn.Module):

    def __init__(self):

        #super() function makes class inheritance more manageable and extensible

        super(Custom_Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size =5,stride= 1, padding = 0, padding_mode = 'zeros')

        self.relu1 = nn.ReLU()

        self.linear1 = nn.Linear(6*28*28,3)



#     def to(device):

#         """Move tensor(s) to chosen device"""

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.relu1(x)

        x = torch.flatten(x,1)

        x = self.linear1(x)

        output = F.log_softmax(x, dim = 1)

        return output

        
from __future__ import print_function

import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

# from Custom_Network import *

from datetime import datetime





def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()



        if batch_idx % 10 ==0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))







def test(model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction = 'sum').item()

            pred = output.argmax(dim = 1, keepdim = True) # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss/=len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))





def save_models(model):

    print()

    torch.save(model.state_dict(), "./trained.model")

    print("****----Checkpoint Saved----****")

    print()







def main():

    

    train_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    

    test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    

    train_dataset = datasets.ImageFolder('../input/training-data-set', train_transform)

    test_dataset = datasets.ImageFolder('../input/category', test_transform)

    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10,

                                              shuffle = True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 2,

                                             shuffle = False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=Custom_Network()

    model=model.to(device)

  

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    scheduler = StepLR(optimizer, step_size = 1, gamma = 0.8)



    # set you own epoch

    for epoch in range(20):



        """

        use train and test function to train and test your model



        """

        train(model, device, train_loader, optimizer, epoch)

        test(model, device, test_loader)

    save_models(model)





if __name__ == "__main__":

    main()