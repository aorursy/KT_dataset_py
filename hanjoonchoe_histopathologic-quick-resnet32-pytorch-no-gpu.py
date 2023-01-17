# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''



# Any results you write to the current directory are saved as output.

import torch

import cv2

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from functools import partial

from sklearn.model_selection import train_test_split

import torch.optim as optim

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import gc
path = '/kaggle/input/histopathologic-cancer-detection'

train = pd.read_csv(os.path.join(path,'train_labels.csv'))
train
class TumorDataset(Dataset):

    def __init__(self,df, kind = 'train'):

        self.df = df

        self.kind = kind

        self.path = '/kaggle/input/histopathologic-cancer-detection'

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        if (self.kind == 'train') or (self.kind =='test'):

            label = self.df.label.values[idx]

            fname = self.df.id.values[idx]

            #Image = cv2.resize(cv2.imread(os.path.join(self.path,self.kind,fname+'.tif')),(224,224))

            Image = cv2.imread(os.path.join(self.path,self.kind,fname+'.tif'))

            return torch.tensor(Image/255.0).permute(2,1,0),label

        else:

            fname = self.df.id.values[idx]

            Image = cv2.imread(os.path.join(self.path,'test',fname+'.tif'))

            return torch.tensor(Image/255.0).permute(2,1,0)
train_set = TumorDataset(train)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=16,shuffle=True)
a = next(iter(train_loader))

fig,ax = plt.subplots(4,4, figsize=(10,10))



for i in range(16):

    j = i//4

    k = i%4

    ax[j,k].imshow(a[0][i].permute(2,1,0))

    ax[j,k].set_title(f'label : {a[1][i]}')

fig.tight_layout()

plt.show()
class Conv2dAuto(nn.Conv2d):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

        

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False) 

def activation_func(activation):

    return  nn.ModuleDict([

        ['relu', nn.ReLU(inplace=True)],

        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],

        ['selu', nn.SELU(inplace=True)],

        ['none', nn.Identity()]

    ])[activation]



class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation='relu'):

        super().__init__()

        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation

        self.blocks = nn.Identity()

        self.activate = activation_func(activation)

        self.shortcut = nn.Identity()   

    

    def forward(self, x):

        residual = x

        if self.should_apply_shortcut: residual = self.shortcut(x)

        x = self.blocks(x)

        x += residual

        x = self.activate(x)

        return x

    

    @property

    def should_apply_shortcut(self):

        return self.in_channels != self.out_channels

    

class ResNetResidualBlock(ResidualBlock):

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):

        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv

        self.shortcut = nn.Sequential(

            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,

                      stride=self.downsampling, bias=False),

            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

        

        

    @property

    def expanded_channels(self):

        return self.out_channels * self.expansion

    

    @property

    def should_apply_shortcut(self):

        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):

    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))



class ResNetBasicBlock(ResNetResidualBlock):

    """

    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation

    """

    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):

        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.blocks = nn.Sequential(

            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),

            activation_func(self.activation),

            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),

        )

class ResNetBottleNeckBlock(ResNetResidualBlock):

    expansion = 4

    def __init__(self, in_channels, out_channels, *args, **kwargs):

        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)

        self.blocks = nn.Sequential(

           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),

             activation_func(self.activation),

             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),

             activation_func(self.activation),

             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),

        )

class ResNetLayer(nn.Module):

    """

    A ResNet layer composed by `n` blocks stacked one after the other

    """

    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):

        super().__init__()

        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'

        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),

            *[block(out_channels * block.expansion, 

                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]

        )



    def forward(self, x):

        x = self.blocks(x)

        return x

    

class ResNetEncoder(nn.Module):

    """

    ResNet encoder composed by layers with increasing features.

    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 

                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):

        super().__init__()

        self.blocks_sizes = blocks_sizes

        

        self.gate = nn.Sequential(

            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),

            nn.BatchNorm2d(self.blocks_sizes[0]),

            activation_func(activation),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )

        

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        self.blocks = nn.ModuleList([ 

            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 

                        block=block,*args, **kwargs),

            *[ResNetLayer(in_channels * block.expansion, 

                          out_channels, n=n, activation=activation, 

                          block=block, *args, **kwargs) 

              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       

        ])

        

        

    def forward(self, x):

        x = self.gate(x)

        for block in self.blocks:

            x = block(x)

        return x

class ResnetDecoder(nn.Module):

    """

    This class represents the tail of ResNet. It performs a global pooling and maps the output to the

    correct class by using a fully connected layer.

    """

    def __init__(self, in_features, n_classes):

        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder = nn.Linear(in_features, n_classes)



    def forward(self, x):

        x = self.avg(x)

        x = x.view(x.size(0), -1)

        x = self.decoder(x)

        return x

class ResNet(nn.Module):

    

    def __init__(self, in_channels, n_classes, *args, **kwargs):

        super().__init__()

        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)

        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

        self.sigmoid = nn.Sigmoid()

        

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        x = self.sigmoid(x)

        return x
def resnet34(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):

    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)
!pip install torchsummary

from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet34(3, 1).to(device)

summary(model, (3, 32, 32))
train_loss = []

valid_loss = []

train_acc = []

valid_acc = []

train_auc = []

valid_auc = []

'''

def criterion(input, target, size_average=True):

    """Categorical cross-entropy with logits input and one-hot target"""

    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)

    if size_average:

        l = l.mean()

    else:

        l = l.sum()

    return l

'''

class Learner(object):

    def __init__(self,df):

        self.df = df

    def fit(self,epochs,batch_size,shuffle):



        self.train(epochs,model,batch_size,shuffle)

    def train(self,epochs,model,batch_size,shuffle):

        train_df, valid_df, _, _ = train_test_split(self.df, self.df['label'],

                                                    stratify=self.df['label'], 

                                                    test_size=0.1)

        train_set = TumorDataset(train_df)

        valid_set = TumorDataset(valid_df)

        train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=shuffle)

        valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=500,shuffle=shuffle)

        

        for epoch in range(epochs):

            model.train()

            running_loss = 0.0

            running_acc = 0.0

            running_auc = 0.0

            print(f'epoch {epoch+1}/{epochs}')

            for idx, (inputs,targets) in tqdm(enumerate(train_loader),total=len(train_loader)):

                optimizer.zero_grad()

                inputs.to(device)

                targets.to(device)

                outputs = model(inputs.float())#.cuda())

                loss = criterion(outputs.cpu().squeeze(),targets.float())

                loss.backward()

                optimizer.step()

                running_loss += loss

                running_acc += (outputs.cpu().round() == targets).float().mean()

                running_auc += roc_auc_score_FIXED(targets.detach().numpy(),outputs.detach().squeeze().numpy())

                if idx%20 == 19:

                    gc.collect()

            train_loss.append(running_loss/len(train_loader))

            train_acc.append(running_acc/len(train_loader))

            train_auc.append(running_auc/len(train_loader))

            print('train loss {:.2f} train acc {:.2f} train auc {:.3f}'.format(running_loss/len(train_loader),running_acc/len(train_loader),running_auc/len(train_loader)))

            if epoch%2 == 1:

                self.valid(valid_loader,model)



    def valid(self,data_loader,model):

        model.eval()

        running_loss = 0.0

        running_acc = 0.0

        running_auc = 0.0

        for idx, (inputs,targets) in enumerate(data_loader):

            with torch.no_grad():

                inputs.to(device)

                targets.to(device)

                outputs = model(inputs.float())#.cuda())

                loss = criterion(outputs.cpu().squeeze(),targets.float())

                running_loss += loss

                running_acc += (outputs.cpu().round() == targets).float().mean()

                running_auc += roc_auc_score_FIXED(targets.detach().numpy(),outputs.detach().squeeze().numpy())

        valid_loss.append(running_loss/len(data_loader))

        valid_acc.append(running_acc/len(data_loader))

        valid_auc.append(running_auc/len(data_loader))

        print('valid loss {:.2f} valid acc {:.2f}'.format(running_loss/len(data_loader),running_acc/len(data_loader)))
gc.collect()
from sklearn.metrics import roc_auc_score, accuracy_score

def roc_auc_score_FIXED(y_true, y_pred):

    if len(np.unique(y_true)) == 1: # bug in roc_auc_score

        return accuracy_score(y_true, np.rint(y_pred))

    return roc_auc_score(y_true, y_pred)
train_zero = train.loc[train.label==0].sample(150)

train_one = train.loc[train.label==1].sample(150)

train_downsample = pd.concat([train_zero,train_one],ignore_index=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet34(3, 1).to(device)

criterion = nn.BCELoss(reduce=True)

optimizer = optim.Adam(model.parameters(),lr = 1e-5)

learner = Learner(train_downsample)

learner.fit(epochs=20,batch_size=15,shuffle=True)
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

axs[0,0].plot(train_loss)

axs[0,0].set_title('Train Loss')

axs[0,1].plot(train_acc)

axs[0,1].set_title('Train acc')

axs[0,2].plot(train_auc)

axs[0,2].set_title('Train auc')

axs[1,0].plot(valid_loss)

axs[1,0].set_title('Valid Loss')

axs[1,1].plot(valid_acc)

axs[1,1].set_title('Valid acc')

axs[1,2].plot(valid_auc)

axs[1,2].set_title('Valid auc')

fig.tight_layout()
test = pd.read_csv(os.path.join(path,'sample_submission.csv'))

test_set = TumorDataset(test,kind='test_no_label')

test_loader = torch.utils.data.DataLoader(test_set,batch_size=300,shuffle=False)

prediction = []

model.eval()

with torch.no_grad():

    for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):

        inputs.to(device)

        outputs = model(inputs.float())#.cuda())

        preds = outputs.detach().cpu().numpy()

        prediction.append(preds)

        

del model

prediction = np.hstack(np.vstack(prediction))



test['label'] = prediction

test.to_csv('submission.csv', index=False)



test_set = TumorDataset(test,kind='test')

test_loader = torch.utils.data.DataLoader(test_set,batch_size=16,shuffle=True)



a = next(iter(test_loader))

fig,ax = plt.subplots(4,4, figsize=(10,10))



for i in range(16):

    j = i//4

    k = i%4

    ax[j,k].imshow(a[0][i].permute(2,1,0))

    ax[j,k].set_title('label : {:.4f}'.format(a[1][i]))

fig.tight_layout()

plt.show()