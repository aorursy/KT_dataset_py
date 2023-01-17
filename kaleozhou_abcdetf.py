# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# coding: utf-8
from __future__ import print_function
from torchvision import transforms,utils
from torchvision import  datasets
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import argparse 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from torch.autograd import Variable 
from PIL import Image
trans=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
class_names=['A', 'B', 'C', 'D', 'E', 'F', 'T', '×', '√']
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 9)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #24*24*20
        x = F.max_pool2d(x, 2, 2) #12*12*50
        x = F.relu(self.conv2(x)) #8*8*50
        x = F.max_pool2d(x, 2, 2) #4*4*50
        # print(x.shape)
        x = x.view(-1, 4*4*50)   #800
        x = F.relu(self.fc1(x))  #500
        x = self.fc2(x)       #10
        return F.log_softmax(x, dim=1) #log_softmax函数 在softmax的基础上 取了对数 为了使用交叉熵函数
#训练函数
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) #若device有cuda ，则使用gpu
        optimizer.zero_grad() #梯度清零
        output = model(data) #前向传播
        loss = F.nll_loss(output, target) #计算交叉熵函数
        loss.backward() #反向传播
        optimizer.step() #参数更新
        if batch_idx % args.log_interval == 0: #能整除log_interval时 打印 相关记录 epoch, batch_index/总的训练样本 ，当前训练数据占总样本比例 ，loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#定义测试函数
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)') #batch-size数目
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 20)') #测试数据的每次大小
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--img', type=str, default='a.jpg',help='test img path')
    parser.add_argument('--train', action='store_true', default=False,help='True train False test')
    #args = parser.parse_args()
    args=parser.parse_known_args()[0]
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_data = datasets.ImageFolder(r'/kaggle/input/lettersabcdetf/Data/train',transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    # print(train_data.classes)
    valid_data=datasets.ImageFolder(r'/kaggle/input/lettersabcdetf/Data/valid',transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
   
   
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)    
    # for i_batch, img in enumerate(train_loader):
    #     if i_batch == 0:
    #         print(img[1])   #标签转化为编码
    #         fig = plt.figure()
    #         grid = utils.make_grid(img[0])
    #         plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #         plt.show()
    #     break
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=False, transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        
    if (args.save_model):
        torch.save(model.state_dict(),"/kaggle/working/data.pth")
main()
