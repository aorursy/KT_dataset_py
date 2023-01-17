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
import pandas as pd

import numpy as np

import torch.nn as nn

import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torchvision import transforms

import argparse



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

traindata_path = "/kaggle/input/fashionmnist/fashion-mnist_train.csv"

testdata_path = "/kaggle/input/fashionmnist/fashion-mnist_test.csv"





class FashionMNIST(Dataset):

    def __init__(self, train_or_test, transform=None):

        self.transform = transform

        fashion_df = pd.read_csv(train_or_test)

        self.labels = fashion_df.label.values

        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        label = self.labels[idx]

        img = Image.fromarray(self.images[idx])



        if self.transform:

            img = self.transform(img)



        return img, label







class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)

        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4*4*50, 1000)

        self.fc2 = nn.Linear(1000, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)



loss_fun = nn.CrossEntropyLoss()



def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_fun(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                       100. * batch_idx / len(train_loader), loss.item()))





def test(args, model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_fun(output, target).item() # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()



    test_loss /= len(test_loader.dataset)



    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))





def main():

    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',

                        help='input batch size for training (default: 32)')

    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',

                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=30, metavar='N',

                        help='number of epochs to train (default: 30)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',

                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',

                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,

                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',

                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=400, metavar='N',

                        help='how many batches to wait before logging training status')



    parser.add_argument('--save-model', action='store_true', default=False,

                        help='For Saving the current Model')

    args = parser.parse_known_args()[0]

    use_cuda = not args.no_cuda and torch.cuda.is_available()



    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = FashionMNIST(traindata_path, transform=transform)

    test_data = FashionMNIST(testdata_path, transform=transform)

    

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)



    model = Net().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)



    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)

        test(args, model, device, test_loader)





if __name__ == '__main__':

    main()




