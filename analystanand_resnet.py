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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import time
from tqdm import tqdm
from tqdm import tqdm_notebook
%matplotlib inline
class NormalBlock(nn.Module):
    vol_expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(NormalBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        self.skip_layer = nn.Sequential()
        if stride != 1 or in_channels != self.vol_expansion*out_channels:
            self.skip_layer = nn.Sequential(
                nn.Conv2d(in_channels, self.vol_expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.vol_expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.batch_norm_1(self.conv_layer1(x)))
        out = self.batch_norm_2(self.conv_layer2(out))
#         out += self.skip_layer(x)
        out = F.relu(out)
        return out
class BottleneckBlock(nn.Module):
    vol_expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        self.conv_layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        self.conv_layer3 = nn.Conv2d(out_channels, self.vol_expansion * out_channels, kernel_size=1, bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(self.vol_expansion * out_channels)

        self.skip_layers = nn.Sequential()
        if stride != 1 or in_channels != self.vol_expansion * out_channels:
            self.skip_layers = nn.Sequential(
                nn.Conv2d(in_channels, self.vol_expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.vol_expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.batch_norm_1(self.conv_layer1(x)))
        out = F.relu(self.batch_norm_2(self.conv_layer2(out)))
        out = self.batch_norm_3(self.conv_layer3(out))
#         out += self.skip_layers(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv_layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.vol_expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.vol_expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_layer1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def ResNet18():
    return ResNet(NormalBlock, [2,2,2,2])

def ResNet34():
    return ResNet(NormalBlock, [3,4,6,3])

def ResNet50():
    return ResNet(BottleneckBlock, [3,4,6,3])

def ResNet101():
    return ResNet(BottleneckBlock, [3,4,23,3])

def ResNet152():
    return ResNet(BottleneckBlock, [3,8,36,3])

# training data transformation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470,0.2435,0.2616))])

# training data loader
train_set = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=2)

# test data transformation
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435,0.2616))])
# test data loader
testset = torchvision.datasets.CIFAR10(root='./kaggle/input', train=False,
                                       download=True,
                                       transform=transform_test)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=2)


def train_model(model, loss_function, optimizer, data_loader):
    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0

    # iterate over the training data

    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)


    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss,
    total_acc))
    return total_loss, total_acc
def test_model(model, loss_function, data_loader):
    # set model in evaluation mode
    model.eval()

    current_loss = 0.0
    current_acc = 0

    # iterate over  the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss,total_acc))

    return total_loss, total_acc
def plot_accuracy(test_accuracy: list,train_accuracy,model_name,ep):
    """Plot accuracy"""
    plt.figure()
    x =  range(1,ep+1)
    plt.plot(x,test_accuracy,color='b',label='Test')
    plt.plot(x,train_accuracy,color='r',label='Train')
    plt.title(model_name)
    # plt.xticks(
    #     [i for i in range(0, len(accuracy))],
    #     [i + 1 for i in range(0, len(accuracy))])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    # plt.savefig('{}.png'.format(model_name))
if __name__ == '__main__':
    modelname_list = ['ResNet18','ResNet34','ResNet50','ResNet101','ResNet152']
    models_list = [ResNet18(),ResNet34(),ResNet50(),ResNet101(),ResNet152()]

    for i in range(0,1):
        start_time = time.time()
        name = modelname_list[i]
        print("Model:",name)
        model = models_list[i]
        # select gpu 0, if available# otherwise fallback to cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # transfer the model to the GPU
        model = model.to(device)
        # loss function
        loss_function = nn.CrossEntropyLoss()
        # We'll optimize all parameters
        optimizer = optim.Adam(model.parameters())

        EPOCHS = 50
        # with tqdm(total=EPOCHS) as pbar:
        test_acc,train_acc = [],[]  # collect accuracy for plotting
        for epoch in range(EPOCHS):
            print('Epoch {}/{}'.format(epoch + 1, EPOCHS))

            train_loss,train_accuracy = train_model(model, loss_function, optimizer, train_loader)
            test_loss, test_accuracy = test_model(model, loss_function, test_loader)
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
              # pbar.update(1)
        # pbar.close()
        # torch.save(model, PATH)
        endtime = time.time() - start_time
        print("Endtime %s seconds",endtime)
        plot_accuracy(test_acc,train_acc,name,EPOCHS)
