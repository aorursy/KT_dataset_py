import torch.nn as nn

import torch.nn.functional as F



# code from https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py

class IdentityPadding(nn.Module):

    def __init__(self, in_channels, out_channels, stride):

        super(IdentityPadding, self).__init__()

        

        self.pooling = nn.MaxPool2d(1, stride=stride)

        self.add_channels = out_channels - in_channels

    

    def forward(self, x):

        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))

        out = self.pooling(out)

        return out

    

    

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):

        super(ResidualBlock, self).__init__()

        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 

                               stride=stride, padding=1, bias=False) 

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)



        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 

                               stride=1, padding=1, bias=False) 

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride

        

        self.dropout = nn.Dropout(0.2)

        

        if down_sample:

            self.down_sample = IdentityPadding(in_channels, out_channels, stride)

        else:

            self.down_sample = None





    def forward(self, x):

        shortcut = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.dropout(out)

        

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.dropout(out)



        if self.down_sample is not None:

            shortcut = self.down_sample(x)



        out += shortcut

        out = self.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, num_layers, block, num_classes=7):

        super(ResNet, self).__init__()

        self.num_layers = num_layers

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 

                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(inplace=True)

        

        # feature map size = 112x112x16

        self.layers_2n = self.get_layers(block, 16, 16, stride=1)

        # feature map size = 56x56x32

        self.layers_4n = self.get_layers(block, 16, 32, stride=2)

        # feature map size = 28x28x64

        self.layers_6n = self.get_layers(block, 32, 64, stride=2)



        # output layers

        self.avg_pool = nn.MaxPool2d(28, stride=1)

        self.fc_out = nn.Linear(64, num_classes)

        

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', 

                                        nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    

    def get_layers(self, block, in_channels, out_channels, stride):

        if stride == 2:

            down_sample = True

        else:

            down_sample = False

        

        layers_list = nn.ModuleList(

            [block(in_channels, out_channels, stride, down_sample)])

            

        for _ in range(self.num_layers - 1):

            layers_list.append(block(out_channels, out_channels))



        return nn.Sequential(*layers_list)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)



        x = self.layers_2n(x)

        x = self.layers_4n(x)

        x = self.layers_6n(x)

        

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc_out(x)

        return x





def resnet():

    block = ResidualBlock

    # total number of layers if 6n + 2. if n is 5 then the depth of network is 32.

    model = ResNet(3, block) 

    return model

from torch.hub import load_state_dict_from_url

from torchvision.models import ResNet



class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):

        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channel, channel // reduction, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),

            nn.Sigmoid()

        )



    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)



def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)





class SEBasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None,

                 *, reduction=16):

        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, 1)

        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SELayer(planes, reduction)

        self.downsample = downsample

        self.stride = stride

        

        self.dropout = nn.Dropout(p=0.5)



    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.se(out)

        

        out = self.dropout(out)

        

        

        if self.downsample is not None:

            residual = self.downsample(x)



            

        out += residual

        out = self.relu(out)



        return out





class SEBottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None,

                 *, reduction=16):

        super(SEBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,

                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(planes * 4, reduction)

        self.downsample = downsample

        self.stride = stride





        

    def forward(self, x):

        residual = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)

        out = self.se(out)

        





        if self.downsample is not None:

            residual = self.downsample(x)



        out += residual

        out = self.relu(out)



        return out





def se_resnet18(num_classes=7):

    """Constructs a ResNet-18 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model





def se_resnet34(num_classes=7):

    """Constructs a ResNet-34 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model





def se_resnet50(num_classes=7, pretrained=False):

    """Constructs a ResNet-50 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    if pretrained:

        model.load_state_dict(load_state_dict_from_url(

            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))

    return model





def se_resnet101(num_classes=7):

    """Constructs a ResNet-101 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model





def se_resnet152(num_classes=7):

    """Constructs a ResNet-152 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch

import torch.optim as optim

from torch.optim import lr_scheduler

import torch.backends.cudnn as cudnn

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader



import argparse

from tensorboardX import SummaryWriter

import pickle

import random

from PIL import Image



print('==> Preparing data..')



cropped_image_list = []

label_list = []

img_path = '/kaggle/input/bts-crop2/cropped/'



with open('/kaggle/input/bts-crop2/result.pickle', 'rb') as f:

    data = pickle.load(f)

    

transforms_train = transforms.Compose([

    transforms.RandomCrop(112, padding=4),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

transforms_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])



#number of train set and test set

num_data=len(data.index)

num_test_data = int(num_data/10)

num_train_data = num_data - num_test_data



classes = ('JM', 'JN', 'JH', 'JK', 'RM', 'VV', 'SG')

class_dict = {'JM': 0,'JN':1, 'JH':2, 'JK':3, 'RM':4, 'VV':5, 'SG':6 }

  

for i in data.index:

    img = Image.open(img_path+data[0][i])

    cropped_image_list.append(img)

    

    label = class_dict[data[1][i]]

    label_list.append(label)

    



#dataset making by random select

dataset_train = []

dataset_test = []



#random select test data index

rand = np.random.choice(np.arange(num_data), num_test_data, replace=False)



# class dataset() and __init__() looks good for calling and managing self variables

# but This time, tried to make simple without class



count_test=0

count_train=0

for i in np.arange(num_data):

    if i in rand:

        img = transforms_test(cropped_image_list[i])

        dataset_test.append([img,label_list[i]])

    else:

        img = transforms_train(cropped_image_list[i])

        dataset_train.append([img,label_list[i]])

        

#dataset_test = np.asarray(dataset_test)

#dataset_train = np.asarray(dataset_train)

class FocalLoss(nn.Module):



    def __init__(self, gamma=0):

        super(FocalLoss, self).__init__()

        self.gamma = gamma

        self.ce = torch.nn.CrossEntropyLoss()



    def forward(self, input, target):

        logp = self.ce(input, target)

        p = torch.exp(-logp)

        loss = (1 - p) ** self.gamma * logp

        return loss.mean()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





exp_id=0

lr = 0.1

batch_size = 16

batch_size_test=2

num_worker=1

resume = None

logdir = '/kaggle/output/'



device = 'cuda' if torch.cuda.is_available() else 'cpu'



train_loader = DataLoader(dataset_train, batch_size=batch_size, 

                          shuffle=True, num_workers=num_worker)

test_loader = DataLoader(dataset_test, batch_size=batch_size_test, 

                         shuffle=False, num_workers=num_worker)



# bts class

classes = ('JM', 'JN', 'JH', 'JK', 'RM', 'VV', 'SG')



print('==> Making model..')



#net = resnet()

net = se_resnet18()

net = net.to(device)

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

print('The number of parameters of model is', num_params)

# print(net)



if resume is not None:

    checkpoint = torch.load('./save_model/' + str(exp_id))

    net.load_state_dict(checkpoint['net'])



#criterion = nn.CrossEntropyLoss()

criterion = FocalLoss(gamma=2.0).to(device)

#criterion = nn.TripletMarginLoss(margin=1.0, p=2.0, 

#                      eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')

optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)



decay_epoch = [2400, 3600]

step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, 

                                 milestones=decay_epoch, gamma=0.1)

writer = SummaryWriter(logdir)





def train(epoch, global_steps):

    net.train()



    train_loss = 0

    correct = 0

    total = 0



    for batch_idx, (inputs, targets) in enumerate(train_loader):

        global_steps += 1

        step_lr_scheduler.step()

        inputs = inputs.to(device)

        

        targets = targets.to(device)

        outputs = net(inputs)

        

        loss = criterion(outputs, targets)

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        train_loss += loss.item()

        _, predicted = outputs.max(1)

        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

        

    acc = 100 * correct / total

    print('train epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(

           epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1), acc))



    writer.add_scalar('log/train error', 100 - acc, global_steps)

    return global_steps





def test(epoch, best_acc, global_steps):

    net.eval()



    test_loss = 0

    correct = 0

    total = 0



    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs = inputs.to(device)

            targets = targets.to(device)

            

            

            outputs = net(inputs)

            loss = criterion(outputs, targets)



            test_loss += loss.item()

            _, predicted = outputs.max(1)

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

    

    acc = 100 * correct / total

    print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(

           epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), acc))



    writer.add_scalar('log/test error', 100 - acc, global_steps)

    

    if acc > best_acc:

        print('==> Saving model..')

        state = {

            'net': net.state_dict(),

            'acc': acc,

            'epoch': epoch,

        }

        if not os.path.isdir('save_model'):

            os.mkdir('save_model')

        torch.save(state, './save_model/ckpt.pth')

        best_acc = acc



    return best_acc





if __name__=='__main__':

    best_acc = 0

    epoch = 0

    global_steps = 0

    

    if resume is not None:

        test(epoch=0, best_acc=0)

    else:

        while True:

            epoch += 1

            global_steps = train(epoch, global_steps)

            best_acc = test(epoch, best_acc, global_steps)

            print('best test accuracy is ', best_acc)

            

            if global_steps >= 4800:

                break







# Any results you write to the current directory are saved as output.