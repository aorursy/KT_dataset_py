import torch

import os

import cv2

import random

import time

import shutil



import numpy as np

import pandas as pd 

from PIL import Image

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.datasets as datasets



import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as f

import torch.autograd as autograd

import torchvision.transforms as transforms

data_path = {'train' : "../input/DataSet/Train Images/Train Images/", 'test' : "../input/DataSet/Test Images/Test Images/"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
indx = '4110'

img = np.asarray(Image.open('../input/DataSet/Train Images/Train Images/Large/clean%s.png'%(indx)))

plt.imshow(img)

plt.show()

class ImageDataset(Dataset):

    def __init__(self, data_path, transform=None):

        self.train_small_data_path = data_path['train'] + "Small/"

        self.train_large_data_path = data_path['train'] + "Large/"

        self.train_small_labels = list(os.listdir(self.train_small_data_path))

        self.train_large_labels = list(os.listdir(self.train_large_data_path))

        self.small_suffix = "ground"

        self.large_suffix = "clean"

        self.test_data_path = data_path['test']

        self.examples_small = len([name for name in os.listdir(self.train_small_data_path)])

        self.examples_large = len([name for name in os.listdir(self.train_large_data_path)])

        self.transform = transform

        self.permutation = list(np.random.permutation(self.examples_large + self.examples_small))

        

    def __len__(self):

        return self.examples_small + self.examples_large

    

    def __getitem__(self, indx, size='small'):

        

        if indx > self.examples_large:

            raise Exception("Index should be smaller than %s".format(examples_large+1))

        

        img_path = (self.train_large_data_path + self.train_large_labels[indx] if size == 'large' else self.train_small_data_path + self.train_small_labels[indx])

        img = Image.open(img_path).convert('RGB')

        

        return img

        

        
'''class ConcatDatasets(Dataset):

    

    def __init__(self, *datasets):

        self.datasets = datasets

    

    def __getitem__(self, i):

        return tuple(d[i] for d in self.datasets)

    

    def __len__(self):

        return min(self.d for d in datasets)

'''
transform_img = transforms.Compose([

    transforms.Pad((0, 120), 0),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

])



inv_transform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
'''data = ImageDataset(data_path, transform_img)

train_size = int(0.9 * len(data))

test_size = int(0.1 * len(data))

print(data) '''
'''train_loader = DataLoader(ConcatDatasets(datasets.ImageFolder(data.train_small_data_path), datasets.ImageFolder(data.train_large_data_path)), batch_size=16, shuffle=True)'''
data = datasets.ImageFolder(root=data_path['train'], transform=transform_img)
a = data.imgs[5998]

b = data.imgs[9999]

print(a, b)
train_size = int(0.9 * len(data))

test_size = int(0.1 * len(data))

print(train_size, test_size)
train_data, validation_data = random_split(data, [train_size+1, test_size])
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

test_loader = DataLoader(validation_data, batch_size=8, shuffle=True)

'''class BasicBlock(nn.Module):

    def __init__(self, channels=64, stride=1, pad=1):

        super(BasicBlock, self).__init__()

        self.channels = channels

        self.pad = pad

        self.stride = stride

    

        self.conv_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, padding=self.pad, stride=self.stride, kernel_size=3)

        self.bn_1 = nn.BatchNorm2d(self.channels)

        self.relu_1 = nn.PReLU()

        

        self.conv_2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, padding=self.pad, stride=self.stride, kernel_size=3)

        self.bn_2 = nn.BatchNorm2d(self.channels)

        self.relu_2 = nn.PReLU()

        

    def forward(self, x):

        identity = x

        x = self.relu_1(self.bn_1(self.conv_1(x)))

        x = self.relu_2(self.bn_2(self.conv_2(x)) + identity)

        return x'''
'''class ConvBnReluBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=1, activation=True):

        super(ConvBnReluBlock, self).__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.padding = padding

        self.stride = stride

        self.kernel_size = kernel_size

        self.activation = activation

        

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        self.bn = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.PReLU()

        

    def forward(self, x):

        if self.activation:

            x = self.relu(self.bn(self.conv(x)))

            

        else:

            x = self.bn(self.conv)

        

        return x'''
'''class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:

            self.norm_layer = nn.BatchNorm2d

        

        self.channels = 64

        

        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=2, stride=2, padding=0)

        self.bn1 = self.norm_layer(self.channels)

        self.relu = nn.PReLU()

        

        self.layer1 = self.make_layers(block, 64, layers[0])

        self.layer2 = self.make_layers(block, 128, layers[1], stride=2)

        self.layer3 = self.make_layers(block, 256, layers[2], stride=2)

        self.layer4 = self.make_layers(block, 512, layers[3], stride=2)

        

        self.fc = nn.Linear(512 , num_classes)

        

    def make_layers(self, block, out_channels, blocks, stride=1):

        

        layers = []

        layers.append(block(self.channels, out_channels, stride))

        self.channels = out_channels

        for _ in range(1, blocks):

            layers.append(block(self.channels, out_channels))

        

        return nn.Sequential(*layers)

    

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))

        

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        

        x = torch.flatten(x, 1)

        x = self.fc(x)

        

        return x

        '''
class Block(nn.Module):

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None,

                 base_width=64, dilation = 1, norm_layer=None):

        super(Block, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        if base_width != 64:

            raise ValueError("Supported Base Width is 64")

        if dilation > 1:

            raise NotImplementedError("Dilation > 1 Not supported")

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)

        self.bn1 = norm_layer(out_planes)

        self.relu = nn.PReLU()

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride)

        self.bn2 = norm_layer(out_planes)

        self.stride = stride

        self.downsample = downsample

    

    def forward(self, x):

        identity = x

        

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        

        out = self.conv2(out)

        out = self.bn2(out)

        

        if self.downsample is not None:

            identity = self.downsample(x)

        

        out = out + identity

        out = self.relu(out)

        

        return out

        

class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None,

                base_width=64, dilation=1, norm_layer=None):

        super(BottleNeck, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        width = int(out_planes * (base_width/64.)) 

        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(width)

        self.relu = nn.PReLU()

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, dilation=1, padding=1)

        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_planes*self.expansion, kernel_size=1)

        self.bn3 = nn.BatchNorm2d(out_planes*self.expansion)

        self.downsample = downsample

        self.stride = stride

    

    def forward(self, x):

        if self.downsample is not None:

            identity = self.downsample(x)

        else:

            identity = x

            

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        

        out = self.conv3(out)

        out = self.bn3(out)

        

        out = out + identity

        out = self.relu(out)

        

        return out
class ResNet(nn.Module):

    

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,

                 width_per_group=64, replace_stride_with_dilation=None,

                 norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer

        

        self.in_planes = 64

        self.dilation = 1

        if replace_stride_with_dilation is None:

            #indicates of replacing 2*2 stride with a dilated conv.

            replace_stride_with_dilation = [False, False, False]

        

        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.in_planes)

        self.relu = nn.PReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])

        self.layer2 = self.make_layer(block, 128, layers[1], stride=2,

                                      dilate=replace_stride_with_dilation[0])

        self.layer3 = self.make_layer(block, 256, layers[2], stride=2,

                                     dilate=replace_stride_with_dilation[1])

        self.layer4 = self.make_layer(block, 512, layers[3], stride=2,

                                     dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512*block.expansion, num_classes)

        

    def make_layer(self, block, out_planes, blocks, stride=1, dilate=False):

        norm_layer = self.norm_layer

        downsample=None

        previous_dilation = self.dilation

        if dilate:

            self.dilation *= stride

            stride = 1

        if stride != 1 or self.in_planes != out_planes*block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.in_planes, out_planes*block.expansion, kernel_size=1, stride=stride),

                norm_layer(out_planes*block.expansion),

            )

        

        layers = []

        layers.append(block(self.in_planes, out_planes, stride, downsample,

                            base_width=self.base_width, dilation=previous_dilation,

                            norm_layer=norm_layer))

        self.in_planes = out_planes*block.expansion

        

        for _ in range(1, blocks):

            layers.append(block(self.in_planes, out_planes,

                                base_width=self.base_width, dilation=self.dilation, 

                                norm_layer=norm_layer))

        

        return nn.Sequential(*layers)

    

    def forward(self, x):

        

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        

        return x   

        
def resnet(arch, block, layers, pretrained, progress, **kwargs):

    model = ResNet(block, layers, **kwargs)

    return model
def resnet50(pretrained=False, progress=True, **kwargs):

    return resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress, **kwargs)
rn = resnet50()
rn.to(device)

rn_optim = optim.Adam(rn.parameters(), lr=0.0002)

loss_fn = nn.CrossEntropyLoss()

for i, data in enumerate(train_loader, 0):

    inputs, label = data

    print(inputs.shape)

    break

print(len(train_loader))
loss_arr = []

loss_epoch_arr = []

epochs = 10

for e in range(epochs):

    for i, data in enumerate(train_loader, 0):

        #tic = time.time()

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        

        rn_optim.zero_grad()

        

        outputs = rn(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        rn_optim.step()

        

        loss_arr.append(loss.item())

        #toc = time.time()

        #print(toc-tic)

    loss_epoch_arr.append(loss.item())

    torch.save({

            'epoch': e,

            'model_state_dict': rn.state_dict(),

            'optimizer_state_dict': rn_optim.state_dict(),

            'loss': loss,

            }, "../input/DataSet/pretrained.pth")



plt.plot(loss_epoch_arr)

plt.show()