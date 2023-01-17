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
import pandas as pd
import numpy as np
import platform

from IPython.display import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import make_grid
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import datasets


import matplotlib.pyplot as plt
%matplotlib inline

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    

transform_train = transforms.Compose(
    [
#    transforms.ToPILImage(), すでにRGBなので、ToPILImage()は不要
#    RandAffine,
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

transform_test = transforms.Compose(
    [
#   transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])



if platform.system() == "Linux":
    data_dir = "../output/kaggle/working"
elif platform.system() == "Darwin":
    data_dir = "./data"
train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
classes = train_dataset.classes
batch_size = 10
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # pytorchは、N（BatchSize）xCxHxWが必要。
    # transpose((1,2,0))で、HxWxCへ戻している。
    # H（高さ） x W（幅） x C色へ変換
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# iterでbatchサイズ（ここでは4）ごとに取り出す。
dataiter = iter(trainloader) 
images, labels = dataiter.next()
## torchvision.utils.make_gridのこと
imshow(make_grid(images))  
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class CIFAR10ResNet(ResNet):
    def __init__(self):
        #super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        # Conv2dのoutput_channel 64 はデフォルト
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2,bias=False)

model = CIFAR10ResNet()
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        
        # TODO:
        # 1. add batch metric (acc1, acc5)
        # 2. add average metric top1=sum(acc1)/batch_idx, top5 = sum(acc5)/batch_idx

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))
def validate(val_loader, model, criterion):
    model.eval()
    loss = 0
    correct = 0
    
    for _, (data, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        
        loss += criterion(output, target).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(val_loader.dataset)
        
    print('\nOn Val set Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(val_loader.dataset),
        100.0 * float(correct) / len(val_loader.dataset)))
def split_dataframe(dataframe=None, fraction=0.9, rand_seed=1):
    df_1 = dataframe.sample(frac=fraction, random_state=rand_seed)
    df_2 = dataframe.drop(df_1.index)
    return df_1, df_2
def prediciton(test_loader, model):
    model.eval()
    test_pred = torch.LongTensor()
    
    for i, data in enumerate(test_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            
        output = model(data)
        
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
        
    return test_pred
transform_train = transforms.Compose(
    [
#    transforms.ToPILImage(), すでにRGBなので、ToPILImage()は不要
#    RandAffine,
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

transform_test = transforms.Compose(
    [
#   transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])


total_epoches = 10
step_size = 5
base_lr = 0.002

#optimizer = optim.Adam(model.parameters(), lr=base_lr)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
for epoch in range(total_epoches):
    print("\nTrain Epoch {}: lr = {}".format(epoch, exp_lr_scheduler.get_lr()[0]))
    if platform.system() == "Linux":
        data_dir = "../output/kaggle/working"
    elif platform.system() == "Darwin":
        data_dir = "./data"
    trainval_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    n_samples = len(trainval_dataset) # n_samples is 50000
    train_size = int(len(trainval_dataset) * 0.8) # train_size is 40000
    val_size = n_samples - train_size # val_size is 10000

    # shuffleしてから分割してくれる.
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)    

    train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)
    validate(val_loader=val_loader, model=model, criterion=criterion)
    exp_lr_scheduler.step()
