import torch

import torch.nn as nn

import torch.nn.functional as F

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

import torchvision.transforms as transforms
import os

CIFAR_DIR = '../input/cifar-10-batches-py'

os.listdir(CIFAR_DIR)
from __future__ import print_function

from PIL import Image

import os

import os.path

import numpy as np

import sys



import pickle



class CIFAR10(torch.utils.data.Dataset):

    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.



    Args:

        root (string): Root directory of dataset where directory

            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.

        train (bool, optional): If True, creates dataset from training set, otherwise

            creates from test set.

        transform (callable, optional): A function/transform that takes in an PIL image

            and returns a transformed version. E.g, ``transforms.RandomCrop``

        target_transform (callable, optional): A function/transform that takes in the

            target and transforms it.

        download (bool, optional): If true, downloads the dataset from the internet and

            puts it in root directory. If dataset is already downloaded, it is not

            downloaded again.



    """

    filename = "cifar-10-python.tar.gz"

    train_list = [

        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],

        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],

        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],

        ['data_batch_4', '634d18415352ddfa80567beed471001a'],

        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],

    ]



    test_list = [

        ['test_batch', '40351d587109b95175f43aff81a1287e'],

    ]

    meta = {

        'filename': 'batches.meta',

        'key': 'label_names',

        'md5': '5ff9c542aee3614f3951f8cda6e48888',

    }

    

    def __init__(self, root, train=True,

                 transform=None, target_transform=None,

                 download=False):



        self.transform = transform

        self.target_transform = target_transform



        self.train = train  # training set or test set



        if self.train:

            downloaded_list = self.train_list

        else:

            downloaded_list = self.test_list



        self.data = []

        self.targets = []

        self.root = root



        # now load the picked numpy arrays

        for file_name, checksum in downloaded_list:

            file_path = os.path.join(self.root, file_name)

            with open(file_path, 'rb') as f:

                entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])

                if 'labels' in entry:

                    self.targets.extend(entry['labels'])

                else:

                    self.targets.extend(entry['fine_labels'])



        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC



        self._load_meta()



    def _load_meta(self):

        path = os.path.join(self.root, self.meta['filename'])

        with open(path, 'rb') as infile:

            data = pickle.load(infile, encoding='latin1')

            self.classes = data[self.meta['key']]

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        

    def __getitem__(self, index):

        """

        Args:

            index (int): Index



        Returns:

            tuple: (image, target) where target is index of the target class.

        """

        img, target = self.data[index], self.targets[index]



        # doing this so that it is consistent with all other datasets

        # to return a PIL Image

        img = Image.fromarray(img)



        if self.transform is not None:

            img = self.transform(img)



        if self.target_transform is not None:

            target = self.target_transform(target)



        return img, target



    def __len__(self):

        return len(self.data)



    def extra_repr(self):

        return "Split: {}".format("Train" if self.train is True else "Test")
'''ResNet-18 Image classfication for cifar-10 with PyTorch 



Author 'Sun-qian'.



'''

class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):

        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(

            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),

            nn.BatchNorm2d(outchannel),

            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(outchannel)

        )

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(

                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(outchannel)

            )



    def forward(self, x):

        out = self.left(x)

        out += self.shortcut(x)

        out = F.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes=10):

        super(ResNet, self).__init__()

        self.inchannel = 64

        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(64),

            nn.ReLU(),

        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)

        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)

        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)

        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)



    def make_layer(self, block, channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]

        layers = []

        for stride in strides:

            layers.append(block(self.inchannel, channels, stride))

            self.inchannel = channels

        return nn.Sequential(*layers)



    def forward(self, x):

        out = self.conv1(x)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out





def ResNet18():

    return ResNet(ResidualBlock)



class CifarData:

    def __init__(self, filenames, need_shuffle):

        all_data = []

        all_labels = []

        for filename in filenames:

            data, labels = load_data(filename)

            all_data.append(data)

            all_labels.append(labels)

                    

        self._data = np.vstack(all_data)

        self._data = self._data / 127.5 - 1.0

        self._labels = np.hstack(all_labels)

        print(self._data.shape)

        print(self._labels.shape)

        self._num_examples = self._data.shape[0]

        self._need_shuffle = need_shuffle

        self._indicator = 0

        if self._need_shuffle:

            self._shuffle_data()

              

    def _shuffle_data(self):

        # [0,1,2,3,4,5] -> [4,5,2,3,0,1] 混排

        p = np.random.permutation(self._num_examples)

        self._data = self._data[p]

        self._labels = self._labels[p]

              

    def get_data(self):

        if self._need_shuffle:

            self._shuffle_data()

        return all_data, all_labels
# 定义是否使用GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#批处理尺寸(batch_size)

BATCH_SIZE = 128      



# 准备数据集并预处理

transform_train = transforms.Compose([

    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32

    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转

    transforms.ToTensor(),

   # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差

])



transform_test = transforms.Compose([

    transforms.ToTensor(),

   # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

])



trainset = CIFAR10(root=CIFAR_DIR, train=True, download=False, transform=transform_train) #训练数据集

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取



testset = CIFAR10(root=CIFAR_DIR, train=False, download=False, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Cifar-10的标签

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# 模型定义-ResNet

net = ResNet18().to(device)



log_set=[]

log_ep_set=[]

acc_ep_set=[]

global_best_acc = [0, 0.0]

# 超参数设置

EPOCH = 48   

pre_epoch = 0  # 定义已经遍历数据集的次数

LR = 0.001        #学习率



# 定义损失函数和优化方式

criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题

# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) 

#优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)



for epoch in range(pre_epoch, EPOCH):

    print('\nEpoch: %d' % (epoch + 1))

    net.train()

    sum_loss = 0.0

    correct = 0.0

    total = 0.0

    for i, data in enumerate(trainloader, 0):

        # 准备数据

        length = len(trainloader)

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()



        # forward + backward

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # 每训练1个batch打印一次loss和准确率

        sum_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += predicted.eq(labels.data).cpu().sum()

        loss = float(sum_loss) / float(i + 1)

        t_acc = 100.0 * float(correct) / float(total)     

        log_set.append([epoch + 1, (i + 1 + epoch * length), loss, t_acc])       

        

    log_ep_set.append([epoch + 1, (i + 1 + epoch * length), loss, t_acc])



    # 每训练完一个epoch测试一下准确率

    print("Waiting Test!")

    with torch.no_grad():

        correct = 0

        total = 0

        best_acc = 0

        for data in testloader:

            net.eval()

            images, labels = data

            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            # 取得分最高的那个类 (outputs.data的索引号)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum()

            acc = 100. * float(correct) / float(total)

            if acc > best_acc:

                best_acc = acc

        print('batch_acc：%.3f%%' % best_acc)

        if best_acc > global_best_acc[1]:

            global_best_acc[1] = best_acc

            global_best_acc[0] = epoch + 1

        acc_ep_set.append([epoch + 1, best_acc])

print("Training Completed, TotalEPOCH=%d" % EPOCH)

print("{!!!!!}EPOCH=%d,best_acc= %.3f%%" % (global_best_acc[0], global_best_acc[1]))
import pandas as pd



global_best_acc_t = [global_best_acc]





log_set_df = pd.DataFrame(log_set, columns=['epoch','iter','loss','acc'])

log_ep_set_df = pd.DataFrame(log_ep_set, columns=['epoch','iter','loss','acc'])

acc_ep_set_df = pd.DataFrame(acc_ep_set, columns=['epoch','acc'])

global_best_acc_df = pd.DataFrame(global_best_acc_t, columns=['epoch','best_acc'])



log_ep_set_df.plot(x='iter', y=['loss'])

log_ep_set_df.plot(x='iter', y=['acc'])

acc_ep_set_df.plot(x='epoch', y=['acc'])



log_set_df.to_csv('./log.csv')

log_ep_set_df.to_csv('./log_ep.csv')

acc_ep_set_df.to_csv('./acc.csv')

global_best_acc_df.to_csv('./best_acc.csv')
