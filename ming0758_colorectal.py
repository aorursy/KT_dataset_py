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
from torchvision.models.resnet import resnet34
import os
import random
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import torch
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np
import os
from torchvision import transforms, datasets
from torch.utils.data import  SubsetRandomSampler
from torch.utils.data import DataLoader
# 对数据集训练集的处理
PIL_transform=transforms.RandomApply([transforms.RandomHorizontalFlip(p=1),transforms.RandomVerticalFlip(p=1),
                                      transforms.RandomRotation(45)],p=0.5)

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 先调整图片大小至256x256
    PIL_transform,
    transforms.RandomCrop((225, 225)),  # 再随机裁剪到224x224
    #transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))  # 归一化，数值是用ImageNet给出的数值
])
 
# 对数据集验证集的处理
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda
cell_dataset = datasets.ImageFolder(root='/kaggle/input/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/',transform=transform_train)

#define dataloader
dataset_loader = DataLoader(cell_dataset,batch_size=16, shuffle=True,num_workers=4)

split1 = int(0.1 * len(cell_dataset))
split2 = int(0.9 * len(cell_dataset))
index_list = list(range(len(cell_dataset)))
np.random.shuffle(index_list) 
test_idx = index_list[:split1]+index_list[split2:]
train_idx=index_list[split1:split2]

## create training and validation sampler objects
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(test_idx)
#trainset=cell_dataset[split1:split2]

## create iterator objects for train and valid datasets
trainloader = DataLoader(cell_dataset, batch_size=16,sampler=tr_sampler,num_workers=4)
validloader = DataLoader(cell_dataset, batch_size=16,sampler=val_sampler,num_workers=4)
print(device)
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
 
def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    model.train()
    train_acc = 0.0
    for batch_idx, (img, label) in enumerate(trainloader):
        image = img.to(device)
        label = label.to(device)
        optimizer2.zero_grad()
        out = model(image)
        #print('out:{}'.format(out))
        #print(out.shape)
        #print('label:{}'.format(label))
        loss = criterion(out, label)
        loss.backward()
        optimizer2.step()
        train_acc = get_acc(out, label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, batch_idx, len(trainloader), loss.mean(), train_acc))
 
 
def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(validloader):
            image = img.to(device)
            label = label.to(device)
            out = model(image)
 
            _, predicted = torch.max(out.data, 1)
 
            total += image.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
    print("Acc: %f " % ((1.0 * correct.numpy()) / total))
 
 
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2,bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer = nn.Linear(512, 8) #加上一层参数修改好的全连接层
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x
if __name__ =='__main__':
    resnet = resnet34(pretrained=True)
    model = Net(resnet)
    model = model.to(device)
    optimizer1 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8, weight_decay=1e-3)  # 设置训练细节
    optimizer2 = optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=5e-4,amsgrad=True)
    scheduler = StepLR(optimizer2, step_size=4,gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(14):
        train(epoch)
        val(epoch)
    torch.save(model, 'modelcatdog1.pth')  # 保存模型
model=torch.load('modelcatdog1.pth')
model.cpu()
classes=('01_TUMOR','02_STROMA','03_COMPLEX','04_LYMPHO','05_DEBRIS','06_MUCOSA','07_ADIPOSE','08_EMPTY')
class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))
with torch.no_grad():
    for data in validloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(8):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(8):
    print("number of correct:",class_correct[i], "number of total:",class_total[i])
    print('Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
