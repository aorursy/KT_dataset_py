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
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

path_list = [
    "/kaggle/input/smotedataset/target.npz",
    "/kaggle/input/smotedataset/data.npz",
    "/kaggle/input/af4500/target_4000.npz",
    "/kaggle/input/af4500/target_5000.npz",
    "/kaggle/input/af4500/data_5000.npz",
    "/kaggle/input/af4500/data_4000.npz"
]


# serial = {"N": 0, "A": 1, "O": 2, "~": 3}
class AFDataSet(Dataset):
    def __init__(self, partition, train=True, dataset_index=0, k=5):
        super().__init__()
        
        # 设置所用数据文件
        if dataset_index == 0:
            data_path = path_list[1]
            target_path = path_list[0]
        elif dataset_index == 1:
            data_path = path_list[5]
            target_path = path_list[2]
        elif dataset_index == 2:
            data_path = path_list[4]
            target_path = path_list[3]

        # 加载数据文件
        data = np.load(data_path)['arr_0']
        target = np.load(target_path)['arr_0']
        # 计算交叉检验分区尺寸
        partition_size = target.shape[0] // k
        
        # 拼接交叉检验分区 ----训练集和测试集
        if train:
            data = np.concatenate(
                (data[:partition * partition_size],
                 data[(partition + 1) * partition_size:]),
                axis=0
            )
            target = np.concatenate(
                (target[:partition * partition_size],
                 target[(partition + 1) * partition_size:]),
                axis=0
            )
        else:
            data = data[partition * partition_size:(partition + 1) * partition_size]
            target = target[partition * partition_size:(partition + 1) * partition_size]


        # 添加通道 满足Pytorch内部Api规范
        self.data = torch.Tensor(data).unsqueeze(1)
        self.target = torch.Tensor(target)
        print(self.data.shape, self.target.shape, Counter(target))

    def __getitem__(self, index):
        temp_data = self.data[index]
        temp_target = self.target[index]
        return temp_data, temp_target

    def __len__(self):
        return len(self.target)
import time
from torch import nn
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# SE块
class SEBlock(nn.Module):
    def __init__(self, in_channel, r=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool1d(1)  # GAP Global Average Pool
        self.se = nn.Sequential(
            nn.Linear(in_channel, in_channel // r, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channel // r, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, length, = x.size()
        out = self.gap(x).view(batch, channel, 1)
        
        # 规整为一维数据 输入到Linear
        out = self.se(out.view(-1, channel))
        
        #规整为二维数据
        out = out.view(batch, channel, 1)
        
        #  规整到原通道尺寸 与原tensor相乘
        return x * out.expand_as(x)

# SE块
class ResidualBlock(nn.Module):
    # 实现子module: Residual Block
    def __init__(self, kernel_size, in_channel, out_channel, padding, shortcut=0):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            nn.BatchNorm1d(out_channel),
            SEBlock(out_channel)
        )
        self.right = shortcut  # Shortcut branch

    def forward(self, x):
        out = self.left(x)
        residual = 0 if self.right == 0 else x
        out += residual
        return F.relu(out)
        
    
class ResNet(nn.Module):
    def __init__(self, block_num, shortcut):
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, stride=1),  #  1x9000
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3)  #  1x3000
        )

        self.layer_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),  # 1x3000
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3)  # 1x1000
        )

        self.layer_3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),  #  1x1000
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3)  #  1x333
        )
        
        self.layer_4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1),  #  1x333
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3)  #  1x111
        )
        

        self.block_layer1 = self.make_layer(kernel_size=3, in_channel=32, out_channel=32,
                                            padding=1, block_num=block_num, shortcut=shortcut)

        self.block_layer2 = self.make_layer(kernel_size=3, in_channel=64, out_channel=64,
                                            padding=1, block_num=block_num, shortcut=shortcut)

        self.block_layer3 = self.make_layer(kernel_size=3, in_channel=128, out_channel=128,
                                            padding=1, block_num=block_num, shortcut=shortcut)
        
        self.block_layer4 = self.make_layer(kernel_size=3, in_channel=256, out_channel=256,
                                            padding=1, block_num=block_num, shortcut=shortcut)
        

        self.fc = nn.Sequential(
            nn.Linear(256 * 111, 6000, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(6000, 1280, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1280, 128, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 4, bias=True),
            nn.Softmax(dim=1)
        )
    
    #添加残差结构
    def make_layer(self, kernel_size, in_channel, out_channel, padding, block_num, shortcut=None):
        layers = list()
        for _ in range(block_num):
            layers.append(ResidualBlock(kernel_size, in_channel, out_channel, padding, shortcut))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.block_layer1(x)
        x = self.layer_2(x)
        x = self.block_layer2(x)
        x = self.layer_3(x)
        x = self.block_layer3(x)
        x = self.layer_4(x)
        x = self.block_layer4(x)

        input_x = x.view(-1, 1 * 256 * 111)
        input_x = self.fc(input_x)
        return input_x

def train(dataset_index=0, batch=128, plist=[0,1,2,3,4]):
    block_num = 4         # “残差结构”
    shortcut = 1          # 是否使用Shotrcut分支

    #  五折交叉检验 plist分区列表 避免长时间会话开启中途因网络问题断开 可手动调整需要检验的数据分区
    for _ in plist:
        train_set = AFDataSet(partition=_, train=True, dataset_index=dataset_index)
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
        
        resnet = ResNet(block_num=block_num, shortcut=shortcut)
        if torch.cuda.is_available():
            resnet.to("cuda:0")

        criterion = nn.CrossEntropyLoss()
        # 含动量的随机梯度下降 weight_decay对应 l2 正则化系数
        optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.7, weight_decay=0.05)
        loss_list = []       # 记录训练中的loss变化

        resnet.train()
        
        #  50 epoch训练
        for epoch in range(50):
            start = time.time()

            running_loss = 0.0
            for i, data in enumerate(train_loader):

                if torch.cuda.is_available():
                    inputs, labels = data[0].to("cuda:0"), data[1].to("cuda:0")
                else:
                    inputs, labels = data

                labels = labels.long()

                optimizer.zero_grad()

                output = resnet(inputs)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %d, %5d] loss: %.4f' % (_, epoch + 1, i + 1, running_loss / 100))
                    loss_list.append(running_loss/100)
                    running_loss = 0.0
        print("Finish training: ", _)
    
        test_set = AFDataSet(partition=_, train=False, dataset_index=dataset_index)
        test_loader = DataLoader(test_set)
        resnet.eval()
        with torch.no_grad():
            class_correct = list(0 for i in range(4))
            class_total = list(0 for i in range(4))
            class_predict = list(0 for i in range(4))
            lab = ["N", "A", "O", "~"]

            for data in test_loader:
                if torch.cuda.is_available():
                    inputs, labels = data[0].to("cuda:0"), data[1].to("cuda:0")
                else:
                    inputs, labels = data
                # inputs, labels = data
                outputs = resnet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                class_total[labels.int()] += 1
                class_predict[predicted.int()] += 1
                if predicted == labels.long():
                    class_correct[labels.int()] += 1
            # 打印 N A O ~ 的识别结果 TP / TP + FN / TP + FP
            print(class_correct[0], class_correct[1], class_correct[2], class_correct[3])
            print(class_total[0], class_total[1], class_total[2], class_total[3])
            print(class_predict[0], class_predict[1], class_predict[2], class_predict[3])

# 参数解释：
# （1）使用的数据集 0-3000x4, 1-4000x4, 2-5000x4
# （2）batch_size
# （3）要检验的分区列表, 默认全使用
train(0, 64, [0, 1, 2, 3,4])