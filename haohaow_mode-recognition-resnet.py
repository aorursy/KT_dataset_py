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
# global_settings.py

import os

os.getcwd()

import os

import argparse

from datetime import datetime

from torchvision import transforms



CHECKPOINT_PATH = 'checkpoint'

NOISE_LEVEL = '_0'



TIME_NOW = datetime.now().isoformat() # time of we run the script

LOG_DIR = 'runs' # tensorboard log dir

SAVE_EPOCH = 10 # save weights file per SAVE_EPOCH epoch

NET_NAME = 'resnet18'

GPU = True



mean = [0.5]; stdv = [0.2]

TRAIN_TRANSFORMS = transforms.Compose([

    transforms.Resize((112, 112), interpolation=2),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(mean=mean, std=stdv)])

TEST_TRANSFORMS = transforms.Compose([

    transforms.Resize((112, 112), interpolation=2),

    transforms.ToTensor(),

    transforms.Normalize(mean=mean, std=stdv)

])

WORKERS = 4

BATCH = 16

WARM = 1

LR = 0.001

MOMENTUM = 0.9

DECAY = 5e-4

EPOCH = 20

MILESTONES = [60, 120, 160]
# utils.py

import os

import shutil

import pandas as pd



if __name__ == '__main__':

    datapath = os.path.join('/kaggle/input/microdoppler/data'+NOISE_LEVEL)  #

    labels = ['spin', 'precess', 'nutation']

    if os.path.exists(os.path.join(os.getcwd(), 'train'+NOISE_LEVEL)) == False:

        os.makedirs(os.path.join(os.getcwd(), 'train'+NOISE_LEVEL))

    if os.path.exists(os.path.join(os.getcwd(), 'val'+NOISE_LEVEL)) == False:

        os.makedirs(os.path.join(os.getcwd(), 'val'+NOISE_LEVEL))

    files_path_train = []

    files_path_test = []

    files_label_train = []

    files_label_test = []



    for label in labels:

        s = 0

        for imgname in os.listdir(os.path.join(datapath,label)):

            if s%7!=0:

                files_path_train.append(imgname)

                files_label_train.append(labels.index(label))

                shutil.copy(os.path.join(datapath, label, imgname),os.path.join(os.getcwd(),'train'+NOISE_LEVEL))



            else:

                files_path_test.append(imgname)

                files_label_test.append(labels.index(label))

                shutil.copy(os.path.join(datapath, label, imgname),os.path.join(os.getcwd(),'val'+NOISE_LEVEL))

            s+=1

    files_train = pd.DataFrame({'name': files_path_train, 'label': files_label_train})

    files_test = pd.DataFrame({'name': files_path_test, 'label': files_label_test})

    print(files_train.loc[0:3])  # [[0,4,7]]

    files_train.to_csv(os.path.join(os.getcwd(),'train'+NOISE_LEVEL+'.txt'))

    files_test.to_csv(os.path.join(os.getcwd(),'val'+NOISE_LEVEL+'.txt'))



# dataloader.py

from PIL import Image

import torch

import os

import pandas as pd

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from torchvision.transforms import ToPILImage





class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset

    def __init__(self, source, transform=None, target_transform=None):

        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化

        self.source = source + NOISE_LEVEL # 将train变成train_0

        self.info = pd.read_csv(

            os.path.join(os.getcwd(), self.source+'.txt'),

            delimiter=',',

            index_col=0)

        self.transform = transform





    def __getitem__(self, idx):  # 对数据进行预处理并返回想要的信息

        file_name, label = self.info.iloc[idx]['name'], self.info.iloc[idx]['label']

        img = Image.open(os.path.join(os.getcwd(),self.source,file_name))

        if self.transform is not None:

            img = self.transform(img)  # 数据标签转换为Tensor

        return img, label



    def __len__(self):

        return len(self.info)

if __name__ == '__main__':

    transform_train = transforms.Compose([transforms.Resize((227,227),interpolation=2),transforms.ToTensor()])

    transform_test = transforms.Compose([transforms.Resize((227,227),interpolation=2),transforms.ToTensor()])

    train_set = MyDataset(source='train',transform=transform_train)

    test_set = MyDataset(source='val',transform=transform_test)

    train_set = DataLoader(train_set, shuffle=True, num_workers=WORKERS, batch_size=BATCH)

    test_set = DataLoader(test_set, shuffle=True, num_workers=WORKERS, batch_size=BATCH)

    for img,idx in test_set:

        print(img.shape,idx.shape)

        img = ToPILImage()(img[0,0,:,:])  # tensor转为PIL Image,[m,C,H,W]

        img.show()

import torch

import torch.nn as nn





class BasicBlock(nn.Module):

    """For resnet 18 and resnet 34

    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()



        # residual function

        self.residual_function = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels * BasicBlock.expansion)

        )



        # shortcut = input

        self.shortcut = nn.Sequential()



        if stride != 1 or in_channels != BasicBlock.expansion * out_channels: # dimension mismatch: H,C,W

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(out_channels * BasicBlock.expansion)

            )



    def forward(self, x):

        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))





class BottleNeck(nn.Module):

    """For resnet over 50 layers

    """

    expansion = 4



    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.residual_function = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channels * BottleNeck.expansion),

        )



        self.shortcut = nn.Sequential()



        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),

                nn.BatchNorm2d(out_channels * BottleNeck.expansion)

            )



    def forward(self, x):

        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))





class ResNet(nn.Module):



    def __init__(self, block, num_block, num_classes=3):

        super().__init__()



        self.in_channels = 64

        self.conv_1 = nn.Conv2d(512 * block.expansion, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, self.in_channels, kernel_size=7, stride=1, padding=1, bias=False),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1) # use a different input_size than the original paper

        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)

        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)

        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.lstm_x = nn.LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)

#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(448, num_classes)



    def _make_layer(self, block, out_channels, num_blocks, stride):



        strides = [stride] + [1] * (num_blocks - 1) # the first block of a layer has a stride of 2，others are equal to 1.

        layers = []

        for stride in strides:

            layers.append(block(self.in_channels, out_channels, stride))

            self.in_channels = out_channels * block.expansion



        return nn.Sequential(*layers)



    def forward(self, x):

        output = self.conv1(x)

        output = self.conv2_x(output)

        output = self.conv3_x(output)

        output = self.conv4_x(output)

        output = self.conv5_x(output)

        output = self.conv_1(output)

        output = torch.squeeze(output) 

        

        print(output.shape)

        output,_ = self.lstm_x(output)

        print(output.shape)

#         output = self.avg_pool(output)

        output = output.contiguous().view(output.size(0), -1)

        output = self.fc(output)



        return output





def resnet18():



    return ResNet(BasicBlock, [2, 2, 2, 2])



def resnet34():



    return ResNet(BasicBlock, [3, 4, 6, 3])



def resnet50():



    return ResNet(BottleNeck, [3, 4, 6, 3])



def resnet101():



    return ResNet(BottleNeck, [3, 4, 23, 3])



def resnet152():



    return ResNet(BottleNeck, [3, 8, 36, 3])

from torch.optim.lr_scheduler import _LRScheduler

# from preprocess import *

# from resnet import resnet152

# from resnet import resnet101

# from resnet import resnet50

# from resnet import resnet34

# from resnet import resnet18

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms

# from global_settings import *

from torch.utils.data import DataLoader

from torch.autograd import Variable



from tensorboardX import SummaryWriter

# import global_settings as settings

import os



def get_network(net_name, use_gpu=True):

    if net_name == 'resnet18':

        net = resnet18().cuda()

    elif net_name == 'resnet34':

        net = resnet34().cuda()

    elif net_name == 'resnet50':

        net = resnet50().cuda()

    elif net_name == 'resnet101':

        net = resnet101().cuda()

    elif net_name == 'resnet152':

        net = resnet152().cuda()

    return net



class WarmUpLR(_LRScheduler):



    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters

        super().__init__(optimizer, last_epoch)



    def get_lr(self):

        """we will use the first m batches, and set the learning rate to base_lr * m / total_iters

        """

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

        print(self.base_lrs)





def train(epoch): # training strategy within a single_epoch

    net.train()

    for batch_index, (images, labels) in enumerate(train_set):

        # WARM-UP, which is recommended in the origin paper of ResNet

        if epoch <= WARM:

            warmup_scheduler.step()

            print('WARMING MODEL!!')



        images = Variable(images)

        labels = Variable(labels)

        images = images.cuda()

        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = net(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()



#         n_iter = (epoch - 1) * len(train_set) + batch_index + 1 # epoch数*每轮batch数+当前batch数



#         last_layer = list(net.children())[-1]

#         for name, para in last_layer.named_parameters():

#             if 'weight' in name:

#                 writer.add_scalar(

#                     'LastLayerGradients/grad_norm2_weights',

#                     para.grad.norm(),

#                     n_iter)

#             if 'bias' in name:

#                 writer.add_scalar(

#                     'LastLayerGradients/grad_norm2_bias',

#                     para.grad.norm(),

#                     n_iter)



#         print(

#             'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(

#                 loss.item(),

#                 optimizer.param_groups[0]['lr'],

#                 epoch=epoch,

#                 trained_samples=batch_index * BATCH + len(images),

#                 total_samples=len(train_set.dataset)))



#         writer.add_scalar('Train/loss', loss.item(), n_iter)



#     for name, param in net.named_parameters():

#         layer, attr = os.path.splitext(name)

#         attr = attr[1:]

#         writer.add_histogram("{}/{}".format(layer, attr), param, epoch)





def eval_training(epoch):

    net.eval()



    test_loss = 0.0  # cost function error

    correct = 0.0



    for (images, labels) in test_set:

        images = Variable(images)

        labels = Variable(labels)



        images = images.cuda()

        labels = labels.cuda()



        outputs = net(images)

        loss = loss_function(outputs, labels)

        test_loss += loss.item()

        _, preds = outputs.max(1)

        correct += preds.eq(labels).sum()



    print('Test set: Average loss: {:.4f}, Acc: {:.4f}'.format(

        test_loss / len(test_set.dataset),

        correct.float() / len(test_set.dataset)

    ))



    # add informations to tensorboard

#     writer.add_scalar('Test/Average loss', test_loss /

#                       len(test_set.dataset), epoch)

#     writer.add_scalar('Test/Adduracy', correct.float() /

#                       len(test_set.dataset), epoch)



    return correct.float() / len(test_set.dataset)





if __name__ == '__main__':



    net = get_network(NET_NAME, use_gpu=GPU)

    # data preprocessing:

    train_set = MyDataset(source='train',  transform=TRAIN_TRANSFORMS)

    test_set = MyDataset(source='val',  transform=TEST_TRANSFORMS)

    train_set = DataLoader(train_set, shuffle=True, num_workers=WORKERS, batch_size=BATCH)

    test_set = DataLoader(test_set, shuffle=True, num_workers=WORKERS, batch_size=BATCH)



    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(

        net.parameters(),

        lr=LR,

        momentum=MOMENTUM,

        weight_decay=DECAY)



    train_scheduler = optim.lr_scheduler.MultiStepLR(

        optimizer, milestones=MILESTONES, gamma=0.2)  # learning rate decay



    iter_per_epoch = len(train_set) # 229，batch_num per iter

    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * WARM)



    checkpoint_path = os.path.join(CHECKPOINT_PATH, NET_NAME, 'lj')



    # create checkpoint folder to save model

    if not os.path.exists(checkpoint_path):

        os.makedirs(checkpoint_path)

    checkpoint_path = os.path.join(checkpoint_path, '{NOISE_LEVEL}_{epoch}_{type}.pth')

    best_acc = 0.0

    for epoch in range(1, EPOCH):

        if epoch > WARM:

            train_scheduler.step(epoch)

        train(epoch)

        acc = eval_training(epoch)



        # start to save best performance model after learning rate decay to

        # 0.01

        if epoch > MILESTONES[1] and best_acc < acc:

            torch.save(

                net.state_dict(),

                checkpoint_path.format(

                    net=NET_NAME,

                    epoch=epoch,

                    type='best'))

            best_acc = acc

            continue



        if not epoch % SAVE_EPOCH:

            torch.save(

                net.state_dict(),

                checkpoint_path.format(

                    net=NET_NAME,

                    epoch=epoch,

                    type='regular'))
!nvidia-smi