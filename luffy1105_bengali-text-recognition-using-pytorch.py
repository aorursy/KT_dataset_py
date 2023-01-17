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
from __future__ import absolute_import, division, print_function



import math

import os

import cv2

import albumentations

import joblib

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo

from PIL import Image

from torch.nn import functional as F

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import models, transforms

from tqdm import tqdm
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

data0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

data1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

data2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

data3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')



data_full = pd.concat([data0, data1, data2, data3], ignore_index=True)

#data_full = joblib.load("data.pkl")

del data0, data1, data2, data3

device = "cuda"

class GraphemeDataset(Dataset):

    def __init__(self, df, label, _type='train'):

        self.df = df

        self.label = label



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        label1 = self.label.vowel_diacritic.values[idx]

        label2 = self.label.grapheme_root.values[idx]

        label3 = self.label.consonant_diacritic.values[idx]

        image = self.df.iloc[idx][1:].values.astype(np.float).reshape(137,236)

        return image, label2, label1, label3

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=dilation, groups=groups, bias=False, dilation=dilation)





def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)





class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:

            raise ValueError(

                'BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:

            raise NotImplementedError(

                "Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None):

        super(Bottleneck, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(inplanes, width)

        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)

        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        identity = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            identity = self.downsample(x)



        out += identity

        out = self.relu(out)



        return out





class ResNet(nn.Module):



    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,

                 groups=1, width_per_group=64, replace_stride_with_dilation=None,

                 norm_layer=None):

        super(ResNet, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer



        self.inplanes = 64

        self.dilation = 1

        if replace_stride_with_dilation is None:

            # each element in the tuple indicates if we should replace

            # the 2x2 stride with a dilated convolution instead

            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:

            raise ValueError("replace_stride_with_dilation should be None "

                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups

        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,

                               bias=False)

        self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,

                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,

                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,

                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, 168)

        self.fc2 = nn.Linear(512 * block.expansion, 11)

        self.fc3 = nn.Linear(512 * block.expansion, 7)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(

                    m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



        # Zero-initialize the last BN in each residual branch,

        # so that the residual branch starts with zeros, and each residual block behaves like an identity.

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:

            for m in self.modules():

                if isinstance(m, Bottleneck):

                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):

                    nn.init.constant_(m.bn2.weight, 0)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        norm_layer = self._norm_layer

        downsample = None

        previous_dilation = self.dilation

        if dilate:

            self.dilation *= stride

            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                conv1x1(self.inplanes, planes * block.expansion, stride),

                norm_layer(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,

                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes, groups=self.groups,

                                base_width=self.base_width, dilation=self.dilation,

                                norm_layer=norm_layer))



        return nn.Sequential(*layers)



    def forward(self, x):

        # See note [TorchScript super()]

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

        p = self.fc1(x)

        q = self.fc2(x)

        r = self.fc3(x)

        return p, q, r





def _resnet(arch, block, layers, pretrained, progress, **kwargs):

    model = ResNet(block, layers, **kwargs)

    if pretrained:

        state_dict = load_state_dict_from_url(model_urls[arch],

                                              progress=progress)

        model.load_state_dict(state_dict)

    return model





def resnet34(pretrained=False, progress=True, **kwargs):

    r"""ResNet-34 model from

    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,

                   **kwargs)
model = resnet34().cuda()

optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)

criterion = nn.CrossEntropyLoss()

batch_size = 32
epochs = 100

model.train()

losses = []

accs = []

reduced_index = train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(

    lambda x: x.sample(5)).image_id.values

reduced_train = train.loc[train.image_id.isin(reduced_index)]

reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]

train_image = GraphemeDataset(reduced_data, reduced_train)

valid_size = 0.2

num_train = len(train_image)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)





train_loader = torch.utils.data.DataLoader(

    train_image, batch_size=batch_size, sampler=train_sampler)

validloader = torch.utils.data.DataLoader(train_image, batch_size=64,

                                          sampler=valid_sampler)

valid_loss_min = np.Inf  # track change in validation loss

model.train()

for epoch in range(epochs):



    print('epochs {}/{} '.format(epoch+1, epochs))

    running_loss = 0.0

    running_acc = 0.0

    running_loss1 = 0.0

    running_acc1 = 0.0

    for idx, (inputs, labels1, labels2, labels3) in tqdm(enumerate(train_loader), total=len(train_loader)):

        inputs = inputs.to(device)

        labels1 = labels1.to(device)

        labels2 = labels2.to(device)

        labels3 = labels3.to(device)



        optimizer.zero_grad()

        outputs1, outputs2, outputs3 = model(inputs.unsqueeze(1).float())

        loss1 = criterion(outputs1, labels1)

        loss2 = criterion(outputs2, labels2)

        loss3 = criterion(outputs3, labels3)

        running_loss += loss1+loss2+loss3

        running_acc += (outputs1.argmax(1) == labels1).float().mean()

        running_acc += (outputs2.argmax(1) == labels2).float().mean()

        running_acc += (outputs3.argmax(1) == labels3).float().mean()

        (loss1+loss2+loss3).backward()

        optimizer.step()



    # scheduler.step()

    model.eval()

    for idx, (inputs, labels1, labels2, labels3) in tqdm(enumerate(validloader), total=len(validloader)):

        inputs = inputs.to(device)

        labels1 = labels1.to(device)

        labels2 = labels2.to(device)

        labels3 = labels3.to(device)



        with torch.no_grad():



            outputs1, outputs2, outputs3 = model(inputs.unsqueeze(1).float())

            loss1 = criterion(outputs1, labels1)

            loss2 = criterion(outputs2, labels2)

            loss3 = criterion(outputs3, labels3)

        running_loss1 += loss1+loss2+loss3

        running_acc1 += (outputs1.argmax(1) == labels1).float().mean()

        running_acc1 += (outputs2.argmax(1) == labels2).float().mean()

        running_acc1 += (outputs3.argmax(1) == labels3).float().mean()



    train_loss = running_loss/len(train_loader)

    valid_loss = running_loss1/len(train_loader)



    print('train acc : {:.4f}%'.format(running_acc/(len(train_loader)*3)))

    print('train loss : {:.4f}'.format(running_loss/len(train_loader)))

    print('valid acc : {:.4f}%'.format(running_acc1/(len(train_loader)*3)))

    print('valid loss : {:.4f}'.format(running_loss1/len(train_loader)))

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

        torch.save(model.state_dict(), 'validation50epoch.pt')

        valid_loss_min = valid_loss

    torch.save(model.state_dict(), 'novalidation50epoch.pt')



torch.save(model.state_dict(), '50resnet34full.pth')

import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(losses)

ax[0].set_title('loss')

ax[1].plot(accs)

ax[1].set_title('acc')
test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class GraphemeDataset(Dataset):

    def __init__(self,df,_type='train'):

        self.df = df

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):

        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(float)

        return image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet34(pretrained=False).to(device)

model.load_state_dict(torch.load('./novalidation50epoch.pt'))
def Resize(df,size=64):

    resized = {} 

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
%%time

model.eval()

test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']

predictions = []

batch_size=1

for fname in test_data:

    data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')

    data = Resize(data)

    test_image = GraphemeDataset(data)

    test_loader = torch.utils.data.DataLoader(test_image,batch_size=1,shuffle=False)

    with torch.no_grad():

        for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):

            inputs.to(device)

            

            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float().cuda())

            predictions.append(outputs3.argmax(1).cpu().detach().numpy())

            predictions.append(outputs2.argmax(1).cpu().detach().numpy())

            predictions.append(outputs1.argmax(1).cpu().detach().numpy())
submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

submission.target = np.hstack(predictions)

submission.head(10)
submission.to_csv('submission.csv',index=False)
