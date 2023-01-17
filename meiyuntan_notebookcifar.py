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
pwd
import time

from tqdm.notebook import trange
from matplotlib import pyplot as plt

import torch
from  torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import googlenet
# 随机数种子
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 2行代码表示每次返回的卷积算法将使用默认算法
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.is_available()
!/opt/bin/nvidia-smi
# 数据加载
train_data = CIFAR10(root='./root/cifar10',
                     train=True,
                     transform=transforms.ToTensor(),
                     download=True)

test_data = CIFAR10(root='./root/cifar10',
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)
type(train_data.data), train_data.data.shape
type(test_data.data), test_data.data.shape
type(train_data[0]), len(train_data[0])
type(train_data[0][0]), type(train_data[0][1])
# 第一张图转为Image
imgage0 = transforms.ToPILImage()(train_data[0][0])
type(imgage0)
imgage0.size, imgage0.mode
# 显示第一张图，此时图像size(32, 32)
plt.imshow(imgage0)
plt.show()
# 调整图像大小为(224, 224)，使用线性差值填充
image0_ = transforms.Resize((224, 224))(imgage0)
type(image0_), image0_.size
plt.imshow(image0_)
plt.show()
# 训练集3个通道的均值
channel_mean = train_data.data.reshape((-1, 3)).mean(axis=0) / 255.0
# 训练集3个通道的标准差
channel_std = train_data.data.reshape((-1, 3)).std(axis=0) / 255.0

train_transform = transforms.Compose([
    # 重置大小
    transforms.Resize((224, 224)), 
     # 在随机位置重新裁剪图片                                   
    transforms.RandomCrop(size=(224, 224), padding=4),   
    # 默认以一半的概率水平（左右）翻转图像                              
    transforms.RandomHorizontalFlip(),    
    # 调整图像亮度、对比度、饱和度、色相
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    # 对每个通道标准化
    transforms.Normalize(channel_mean, channel_std),   
    ])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(channel_mean, channel_std),
    ])
channel_mean, channel_std
# 数据加载
train_data = CIFAR10(root='./root/cifar10',
                     train=True,
                     transform=train_transform,
                     download=False)
test_data = CIFAR10(root='./root/cifar10',
                    train=False,
                    transform=test_transform,
                    download=False)
# 超参数
BATCH_SIZE = 128

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
# pretrained 如果为真，则返回在ImageNet上预训练的模型
net_model = googlenet(pretrained=True)
# 可知在Googlenet只有一层全连接层
print(net_model.fc)
# for param in net_model.parameters():
#     print(type(param))
#     print(param.requires_grad)
#     break
# i = 0
# for param in net_model.parameters():
#     param.requires_grad = False
#     i += 1
# print(f'{i} 个参数已全部固定')
# for param in net_model.parameters():
#     print(type(param))
#     print(param.requires_grad)
#     break
net_model.fc = nn.Linear(1024, 10, bias=True)
net_model.fc
# 超参数 学习率
LR = 1e-4

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器中只添加fc层的参数
optimizer = optim.Adam(net_model.parameters(), lr=LR, weight_decay=0.)
# 定义device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_model.to(device)
print(device)
for t in train_loader:
    print(type(t))
    print(len(t))
    print(type(t[0]), type(t[1]))
    print(t[0].shape, t[1].shape)
    break
EPOCH = 10

costs = []
train_accs = []
epoch_bar = trange(EPOCH)
early_stop = 0
min_loss = np.inf

for epoch in epoch_bar:
    epoch_bar.set_description("epoch:{}".format(epoch))
    start = time.time()
    losses = []
    correct, total = 0, 0
    for i, data in enumerate(train_loader):
        feats, labels = data
        feats, labels = feats.to(device), labels.to(device)
        # 前向传播 等价于net_model(feats)
        outputs = net_model.forward(feats)
        # 计算损失函数
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 预测
        _, pred = torch.max(outputs.data, 1)
        # 判断预测与实际是否一致
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    # 计算每个epoch的准确率
    train_accs.append(correct / total)
    batch_mean_loss = np.mean(losses)
    costs.append(batch_mean_loss)
    if batch_mean_loss < min_loss:
        min_loss = batch_mean_loss
        early_stop = 0
    else:
        early_stop += 1
        if early_stop == 10:
            print(
                f"epoch:{epoch} loss:{batch_mean_loss} train_accuracy:{train_accs[-1]} 连续{early_stop}个Epoch未减小，停止循环"
            )
            break
#     if epoch % 5 == 0:
#         print("epoch:{} loss:{:.4f} train_accuracy:{} 耗时:{:.1f}s/epoch".format(epoch, batch_mean_loss, train_accs[-1], time.time() - start))
    print("epoch:{} loss:{:.4f} train_accuracy:{} 耗时:{:.1f}s/epoch".format(
        epoch, batch_mean_loss, train_accs[-1],
        time.time() - start))
plt.plot(range(len(costs)), costs)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss Curve")
plt.show()
plt.plot(range(len(train_accs)), train_accs)
plt.xlabel('Epoch')
plt.ylabel('Train ACC')
plt.title("Train ACC Curve")
plt.show()
torch.save(net_model, "./cifar10_googlenet_1.pt")
# 加载模型的命令
# model = torch.load("cifar10_googlenet.pt")
net_model.eval()

with torch.no_grad():
    correct, total = 0, 0
    for data in test_loader:
        feats, labels = data
        feats, labels = feats.to(device), labels.to(device)
        # 前向传播
        out = net_model(feats)
        # 预测
        _, pred = torch.max(out.data, 1)
        # 判断预测与实际是否一致
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
    print("在测试集10000张图像上的准确率:{:.2f}%".format(correct / total *100))
