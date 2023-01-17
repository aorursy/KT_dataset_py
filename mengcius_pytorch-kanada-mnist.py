# !wget https://mengcius.coding.net/api/share/download/ad2a6c94-1036-409a-a4bf-9cef4088e990 -O ./Kannada-MNIST.zip

# !unzip Kannada-MNIST.zip
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



import torch

import torchvision

from torchvision import transforms, datasets

import torch.nn.functional as F

import torch.nn as nn

import torch.optim as optim

import math

import random

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

print(torch.__version__)
# Load Data

train=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test=pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

submission_set = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv").iloc[:,1:]



train_data=train.drop('label',axis=1)

train_targets=train['label']



test_images=test.drop('label',axis=1)

test_labels=test['label']



# Train Test Split

train_images, val_images, train_labels, val_labels = train_test_split(train_data, 

                                                                     train_targets, 

                                                                     test_size=0.2)



# Reset Index

train_images.reset_index(drop=True, inplace=True)

train_labels.reset_index(drop=True, inplace=True)



val_images.reset_index(drop=True, inplace=True)

val_labels.reset_index(drop=True, inplace=True)



test_images.reset_index(drop=True, inplace=True)

test_labels.reset_index(drop=True, inplace=True)



print("Train Set")

print(train_images.shape)

print(train_labels.shape)



print("Validation Set")

print(val_images.shape)

print(val_labels.shape)



print("Validation 2")

print(test_images.shape)

print(test_labels.shape)



print("Submission")

print(submission_set.shape)
print("Look at image means")

print(train_images.mean(axis = 1).mean())

print(val_images.mean(axis = 1).mean())

print(test_images.mean(axis = 1).mean())

print(submission_set.mean(axis = 1).mean())
print("Train Distribution")

print(train_labels.value_counts(normalize = True))



print("\nSubmission Distribution")

print(test_labels.value_counts(normalize = True))
IMGSIZE = 28



# Transformations for the train

train_trans = transforms.Compose(([

    transforms.ToPILImage(),

    transforms.RandomCrop(IMGSIZE),

    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)), # 保持图像中心不变的随机仿射变换

#     transforms.RandomRotation(10), # 随机旋转(-degrees， +degrees)

#     transforms.RandomErasing(p=0.5, scale=(0.01, 0.02), ratio=(0.3, 3.3)), # 随机擦除

#     transforms.ColorJitter(), # 随机改变图像的亮度、对比度和饱和度

    transforms.ToTensor(), # divides by 255

  #  transforms.Normalize((0.5,), (0.5,))

]))



# Transformations for the validation & test sets

val_trans = transforms.Compose(([

    transforms.ToPILImage(),

    transforms.ToTensor(), # divides by 255

   # transforms.Normalize((0.1307,), (0.3081,))

]))



class KannadaDataSet(torch.utils.data.Dataset):

    def __init__(self, images, labels,transforms = None):

        self.X = images

        self.y = labels

        self.transforms = transforms

         

    def __len__(self):

        return (len(self.X))

    

    def __getitem__(self, i):

        data = self.X.iloc[i,:]

        data = np.array(data).astype(np.uint8).reshape(IMGSIZE,IMGSIZE,1)

        

        if self.transforms:

            data = self.transforms(data)

            

        if self.y is not None:

            return (data, self.y[i])

        else:

            return data
batch_size = 128



train_data = KannadaDataSet(train_images, train_labels, train_trans)

val_data = KannadaDataSet(val_images, val_labels, val_trans)

test_data = KannadaDataSet(test_images, test_labels, val_trans)

submission_data = KannadaDataSet(submission_set, None, val_trans)





train_loader = torch.utils.data.DataLoader(train_data, 

                                           batch_size=batch_size, 

                                           shuffle=True)



val_loader = torch.utils.data.DataLoader(val_data, 

                                           batch_size=batch_size, 

                                           shuffle=False)



test_loader = torch.utils.data.DataLoader(test_data,  # Dig-MNIST.csv

                                          batch_size=batch_size, 

                                          shuffle=False)



submission_loader = torch.utils.data.DataLoader(submission_data,

                                          batch_size=batch_size, 

                                          shuffle=False)



classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
class DenseNet(nn.Module):

    def __init__(self, dropout = 0.40):

        super(DenseNet, self).__init__()

        self.dropout = dropout

        

        # https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch

        #Our batch shape for input x is (1, 28, 28)

        # (Batch, Number Channels, height, width).

        #Input channels = 1, output channels = 18

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.conv1_bn = nn.BatchNorm2d(num_features=64)

        

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.conv1_1_bn = nn.BatchNorm2d(num_features=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.d2_1 = nn.Dropout2d(p=self.dropout)

        

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv2_bn = nn.BatchNorm2d(num_features=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.d2_2 = nn.Dropout2d(p=self.dropout)

        

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv3_bn = nn.BatchNorm2d(num_features=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.d2_3 = nn.Dropout2d(p=self.dropout)

        

        #4608 input features, 256 output features (see sizing flow below)

        self.fc1 = nn.Linear(256 * 3 * 3, 512) # Linear 1

        self.d1_1 = nn.Dropout(p=self.dropout)

        #64 input features, 10 output features for our 10 defined classes

        self.fc2 = nn.Linear(in_features=512, out_features=256) # linear 2

        self.d1_2 = nn.Dropout(p=self.dropout)

        self.fc3 = nn.Linear(in_features=256, out_features=128) # linear 3

        self.d1_3 = nn.Dropout(p=self.dropout)

        self.out = nn.Linear(in_features=128, out_features=10) # linear 3

        

    def forward(self, x):

        #Computes the activation of the first convolution

        #Size changes from (1, 28, 28) to (18, 28, 28)

        x = self.conv1(x)

        x = self.conv1_bn(x)

        x = F.relu(x)

        x = self.conv1_1(x)

        x = self.conv1_1_bn(x)

        x = F.relu(x)       

        

        x = self.d2_1(x)

        x = self.pool1(x) # Size changes from (18, 28, 28) to (18, 14, 14)

        

        # Second Conv       

        x = self.conv2(x)

        x = self.conv2_bn(x)

        x = F.relu(x)

        x = self.d2_2(x)

        x = self.pool2(x) # Size changes from (18, 14, 14) to (18, 7, 7)

        

        # Third Conv       

        x = self.conv3(x)

        x = self.conv3_bn(x)

        x = F.relu(x)

        x = self.d2_3(x)

        x = self.pool3(x) # Size changes from (18, 7, 7) to (18, 3, 3)

        

        #Reshape data to input to the input layer of the neural net

        #Size changes from (18, 14, 14) to (1, 3528)

        #Recall that the -1 infers this dimension from the other given dimension

        x = x.view(-1, 256 * 3 * 3)



        #Computes the activation of the first fully connected layer

        #Size changes from (1, 4608) to (1, 64)

        #Computes the second fully connected layer (activation applied later)

        #Size changes from (1, 64) to (1, 10)

        x = F.relu(self.fc1(x))

        x = self.d1_1(x)

        

        x = F.relu(self.fc2(x))

        x = self.d1_2(x)

        

        x = F.relu(self.fc3(x))

        x = self.d1_3(x)

        

        x = self.out(x)

        return F.log_softmax(x, dim=-1)



# net = DenseNet().to(device)

# net
def outputSize(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1

    return(output)

# outputSize(64, 5, 1, 2)
'''ResNet in PyTorch.



For Pre-activation ResNet, see 'preact_resnet.py'.



Reference:

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    Deep Residual Learning for Image Recognition. arXiv:1512.03385

'''

import torch

import torch.nn as nn

import torch.nn.functional as F





class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(self.expansion*planes)

            )



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = F.relu(out)

        return out





class Bottleneck(nn.Module):

    expansion = 4



    def __init__(self, in_planes, planes, stride=1):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(self.expansion*planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(self.expansion*planes)

            )



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)

        out = F.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):

        super(ResNet, self).__init__()

        self.in_planes = 64



        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)

        layers = []

        for stride in strides:

            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out





def res_net18():

    return ResNet(BasicBlock, [2,2,2,2])



def res_net34():

    return ResNet(BasicBlock, [3,4,6,3])



def res_net50():

    return ResNet(Bottleneck, [3,4,6,3])



def res_net101():

    return ResNet(Bottleneck, [3,4,23,3])



def res_net152():

    return ResNet(Bottleneck, [3,8,36,3])





def test():

    net = res_net18()#.to(device)

    y = net(torch.randn(1,1,28,28))

    print(y.size())



test()
import math

import torch

from torch import nn



try:

    from torch.hub import load_state_dict_from_url

except ImportError:

    from torch.utils.model_zoo import load_url as load_state_dict_from_url



model_urls = {

    'efficientnet_b0': 'https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1',

    'efficientnet_b1': 'https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1',

    'efficientnet_b2': 'https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1',

    'efficientnet_b3': 'https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1',

    'efficientnet_b4': 'https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1',

    'efficientnet_b5': 'https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1',

    'efficientnet_b6': None,

    'efficientnet_b7': None,

}



params = {

    'efficientnet_b0': (1.0, 1.0, 224, 0.2),

    'efficientnet_b1': (1.0, 1.1, 240, 0.2),

    'efficientnet_b2': (1.1, 1.2, 260, 0.3),

    'efficientnet_b3': (1.2, 1.4, 300, 0.3),

    'efficientnet_b4': (1.4, 1.8, 380, 0.4),

    'efficientnet_b5': (1.6, 2.2, 456, 0.4),

    'efficientnet_b6': (1.8, 2.6, 528, 0.5),

    'efficientnet_b7': (2.0, 3.1, 600, 0.5),

}





class Swish(nn.Module):



    def __init__(self, *args, **kwargs):

        super(Swish, self).__init__()



    def forward(self, x):

        return x * torch.sigmoid(x)





class ConvBNReLU(nn.Sequential):



    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):

        padding = self._get_padding(kernel_size, stride)

        super(ConvBNReLU, self).__init__(

            nn.ZeroPad2d(padding),

            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),

            nn.BatchNorm2d(out_planes),

            Swish(),

        )



    def _get_padding(self, kernel_size, stride):

        p = max(kernel_size - stride, 0)

        return [p // 2, p - p // 2, p // 2, p - p // 2]





class SqueezeExcitation(nn.Module):



    def __init__(self, in_planes, reduced_dim):

        super(SqueezeExcitation, self).__init__()

        self.se = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(in_planes, reduced_dim, 1),

            Swish(),

            nn.Conv2d(reduced_dim, in_planes, 1),

            nn.Sigmoid(),

        )



    def forward(self, x):

        return x * self.se(x)





class MBConvBlock(nn.Module):



    def __init__(self,

                 in_planes,

                 out_planes,

                 expand_ratio,

                 kernel_size,

                 stride,

                 reduction_ratio=4,

                 drop_connect_rate=0.2):

        super(MBConvBlock, self).__init__()

        self.drop_connect_rate = drop_connect_rate

        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]

        assert kernel_size in [3, 5]



        hidden_dim = in_planes * expand_ratio

        reduced_dim = max(1, int(in_planes / reduction_ratio))



        layers = []

        # pw

        if in_planes != hidden_dim:

            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]



        layers += [

            # dw

            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),

            # se

            SqueezeExcitation(hidden_dim, reduced_dim),

            # pw-linear

            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),

            nn.BatchNorm2d(out_planes),

        ]



        self.conv = nn.Sequential(*layers)



    def _drop_connect(self, x):

        if not self.training:

            return x

        keep_prob = 1.0 - self.drop_connect_rate

        batch_size = x.size(0)

        random_tensor = keep_prob

        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)

        binary_tensor = random_tensor.floor()

        return x.div(keep_prob) * binary_tensor



    def forward(self, x):

        if self.use_residual:

            return x + self._drop_connect(self.conv(x))

        else:

            return self.conv(x)





def _make_divisible(value, divisor=8):

    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)

    if new_value < 0.9 * value:

        new_value += divisor

    return new_value





def _round_filters(filters, width_mult):

    if width_mult == 1.0:

        return filters

    return int(_make_divisible(filters * width_mult))





def _round_repeats(repeats, depth_mult):

    if depth_mult == 1.0:

        return repeats

    return int(math.ceil(depth_mult * repeats))





class EfficientNet(nn.Module):



    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=10):

        super(EfficientNet, self).__init__()



        # yapf: disable

        settings = [

            # t,  c, n, s, k

            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112

            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56

            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28

            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14

            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14

            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7

            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7

        ]

        # yapf: enable



        out_channels = _round_filters(32, width_mult)

        features = [ConvBNReLU(1, out_channels, 3, stride=1)] # (3, out_channels, 3, stride=2)



        in_channels = out_channels

        for t, c, n, s, k in settings:

            out_channels = _round_filters(c, width_mult)

            repeats = _round_repeats(n, depth_mult)

            for i in range(repeats):

                stride = s if i == 0 else 1

                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]

                in_channels = out_channels



        last_channels = _round_filters(1280, width_mult)

        features += [ConvBNReLU(in_channels, last_channels, 1)]



        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(

            nn.Dropout(dropout_rate),

            nn.Linear(last_channels, num_classes),

        )



        # weight initialization

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out')

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.ones_(m.weight)

                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):

                fan_out = m.weight.size(0)

                init_range = 1.0 / math.sqrt(fan_out)

                nn.init.uniform_(m.weight, -init_range, init_range)

                if m.bias is not None:

                    nn.init.zeros_(m.bias)



    def forward(self, x):

        x = self.features(x)

        x = x.mean([2, 3])

        x = self.classifier(x)

        return x





def _efficientnet(arch, pretrained, progress, **kwargs):

    width_mult, depth_mult, _, dropout_rate = params[arch]

    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)

    if pretrained:

        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)



        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:

            del state_dict['classifier.1.weight']

            del state_dict['classifier.1.bias']



        model.load_state_dict(state_dict, strict=False)

    return model





def efficientnet_b0(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b0', pretrained, progress, **kwargs)



def efficientnet_b1(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b1', pretrained, progress, **kwargs)



def efficientnet_b2(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b2', pretrained, progress, **kwargs)



def efficientnet_b3(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b3', pretrained, progress, **kwargs)



def efficientnet_b4(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b4', pretrained, progress, **kwargs)



def efficientnet_b5(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b5', pretrained, progress, **kwargs)



def efficientnet_b6(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b6', pretrained, progress, **kwargs)



def efficientnet_b7(pretrained=False, progress=True, **kwargs):

    return _efficientnet('efficientnet_b7', pretrained, progress, **kwargs)





def test1():

    net = efficientnet_b0()#.to(device)

    y = net(torch.randn(2,1,28,28))

    print(y.size())





# test1()
import math

import torch

from torch.optim.optimizer import Optimizer, required



class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        step_size = group['lr'] / (1 - beta1 ** state['step'])

                    buffered[2] = step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                if N_sma >= 5:                    

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                else:

                    p_data_fp32.add_(-step_size, exp_avg)



                p.data.copy_(p_data_fp32)



        return loss

    

iter_idx = 0    

class AdamW(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0, use_variance=True, warmup = 4000):

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay, use_variance=True, warmup = warmup)

        print('======== Warmup: {} ========='.format(warmup))

        super(AdamW, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(AdamW, self).__setstate__(state)



    def step(self, closure=None):

        global iter_idx

        iter_idx += 1

        grad_list = list()

        mom_list = list()

        mom_2rd_list = list()



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                state['step'] += 1



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                

                if group['warmup'] > state['step']:

                    scheduled_lr = 1e-6 + state['step'] * (group['lr'] - 1e-6) / group['warmup']

                else:

                    scheduled_lr = group['lr']



                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)



                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)



                p.data.copy_(p_data_fp32)



        return loss
from torch.autograd import Variable



class FocalLoss(nn.Module):

    def __init__(self,reduction='mean',alpha=0.01,gamma=1):

        super(FocalLoss,self).__init__()

        self.reduction = reduction

        self.alpha = alpha

        self.gamma = gamma



    def forward(self,ypred,ytrue):

        logpt = F.log_softmax(ypred,1) 

        pt = Variable(torch.exp(logpt))

        ytrue = to_categorical(ytrue)

        pt_prime = 1 - pt

        focalloss = -self.alpha * (pt_prime)**self.gamma * ytrue * logpt

        focalloss = torch.sum(focalloss,1)

        if(self.reduction=='sum'):

            return focalloss.sum()

        else:

            return focalloss.mean()



numclasses = 10



def to_categorical(ytrue):

    input_shape = ytrue.size()

    n = ytrue.size(0)

    categorical = torch.zeros(n, numclasses).to(device)

    categorical[torch.arange(n), ytrue] = 1

    output_shape = input_shape + (numclasses,)

    categorical = torch.reshape(categorical, output_shape)

    return categorical
class CrossEntropyLabelSmooth(nn.Module):

    

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):

        super(CrossEntropyLabelSmooth, self).__init__()

        self.num_classes = num_classes

        self.epsilon = epsilon

        self.use_gpu = use_gpu

        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, inputs, targets):

        log_probs = self.logsoftmax(inputs)

        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        if self.use_gpu: targets = targets.cuda()

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (- targets * log_probs).mean(0).sum()

        return loss
# Learning Rate Finder https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

def find_lr(trn_loader, init_value = 1e-8, final_value=10., beta = 0.98):

    num = len(trn_loader)-1

    mult = (final_value / init_value) ** (1/num)

    lr = init_value

    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.

    best_loss = 0.

    batch_num = 0

    losses = []

    log_lrs = []

    for data in trn_loader:

        batch_num += 1

        #As before, get the loss for this mini-batch of inputs/outputs

        inputs = data[0].to(device)

        labels = data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        #Compute the smoothed loss

        avg_loss = beta * avg_loss + (1-beta)*loss.item()

        smoothed_loss = avg_loss / (1 - beta**batch_num)

        #Stop if the loss is exploding

        if batch_num > 1 and smoothed_loss > 4 * best_loss:

            return log_lrs, losses

        #Record the best loss

        if smoothed_loss < best_loss or batch_num==1:

            best_loss = smoothed_loss

        #Store the values

        losses.append(smoothed_loss)

        log_lrs.append(math.log10(lr))

        #Do the SGD step

        loss.backward()

        optimizer.step()

        #Update the lr for the next step

        lr *= mult

        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses
# net = DenseNet().to(device)

net = res_net18().to(device)

# net = efficientnet_b0().to(device)



# Loss Function

criterion = nn.CrossEntropyLoss()

# criterion = F.nll_loss

# criterion = FocalLoss(alpha=1,gamma=0)

# criterion = CrossEntropyLabelSmooth(num_classes=10)



# Gradient Descent

# optimizer = optim.SGD(net.parameters(),lr=1e-1)

optimizer = optim.Adam(net.parameters(), lr=1e-4)

# optimizer = RAdam(net.parameters(), lr=1e-1, betas=(0.9, 0.999))

# optimizer = AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999), warmup = 4000)

# optimizer = optim.Adadelta(net.parameters(), lr=0.1)



%matplotlib inline

logs,losses = find_lr(trn_loader = train_loader)

plt.plot(logs[10:-5],losses[10:-5])
torch.manual_seed(1234) 

torch.cuda.manual_seed(1234)

torch.cuda.manual_seed_all(1234)  # multi-GPU

np.random.seed(1234)  # Numpy module

random.seed(1234)  # Python random module

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True



EPOCHS = 80

nn_output = []



# net = DenseNet().to(device)

net = res_net18().to(device)

# net = efficientnet_b0().to(device)



# optimizer = optim.SGD(net.parameters(),lr=1e-2)

optimizer = optim.Adam(net.parameters(), lr=1e-4)

# optimizer = RAdam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

# optimizer = AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999), warmup = 4000)

# optimizer = optim.Adadelta(net.parameters(), lr=0.1)



# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS//4, gamma=0.1)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min = 0)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5) # ReduceLROnPlateau里没有scheduler.get_lr()

# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 0) # torch===1.3.0



criterion = nn.CrossEntropyLoss()

# criterion = F.nll_loss

# criterion = FocalLoss(alpha=1,gamma=0)

# criterion = CrossEntropyLabelSmooth(num_classes=10)





def get_num_correct(preds, labels):

    return preds.argmax(dim=1).eq(labels).sum().item()



for epoch in range(EPOCHS):

    epoch_loss = 0

    epoch_correct = 0

    net.train()

    

    for data in train_loader:

        # `data` is a batch of data

        # Before using transforms, I used .unsqueeze(1) to enter a empty number channel array (Batch, Number Channels, height, width).

        X = data[0].to(device) # X is the batch of features

        # Unsqueeze adds a placeholder dimension for the color channel - (8, 28, 28) to (8, 1, 28, 28)

        y = data[1].to(device) # y is the batch of targets.

        

        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.

        output = net(X)  # pass in the reshaped batch (recall they are 28x28 atm)

        tloss = criterion(output, y)  # calc and grab the loss value

        tloss.backward()  # apply this loss backwards thru the network's parameters

        optimizer.step()  # attempt to optimize weights to account for loss/gradients 

        

        epoch_loss += tloss.item()

        epoch_correct += get_num_correct(output, y)

    print('LR =',scheduler.get_lr())    

    scheduler.step() 

#     lr = optimizer.param_groups[0]['lr'] # ReduceLROnPlateau

#     print('epoch',epoch,',lr',lr)

#     scheduler.step(tloss)   



    # Evaluation with the validation set

    net.eval() # eval mode

    val_loss = 0

    val_correct = 0

    test_loss = 0

    test_correct = 0

    

    with torch.no_grad():

        # First Validation Set

        for data in val_loader:

            X = data[0].to(device)

            y = data[1].to(device)

            

            preds = net(X) # get predictions

            vloss = criterion(preds, y) # calculate the loss

            

            val_correct += get_num_correct(preds, y)

            val_loss += vloss.item()

        

        # Second Validation Set.. Dig-MNIST.csv

        for data in test_loader:

            X = data[0].to(device)

            y = data[1].to(device)

            

            preds = net(X) # get predictions

            tstloss = criterion(preds, y) # calculate the loss

            

            test_correct += get_num_correct(preds, y)

            test_loss += tstloss.item()

    

    tmp_nn_output = [epoch + 1,EPOCHS,

                     epoch_loss/len(train_loader.dataset),epoch_correct/len(train_loader.dataset)*100,

                     val_loss/len(val_loader.dataset), val_correct/len(val_loader.dataset)*100,

                     test_loss/len(test_loader.dataset), test_correct/len(test_loader.dataset)*100

                    ]

    nn_output.append(tmp_nn_output)

    

    # Print the loss and accuracy for the validation set

    print('Epoch[{}/{}] Train loss: {:.6f} acc: {:.3f} | Valid loss: {:.6f} acc: {:.3f} | DigTest loss: {:.6f} acc: {:.3f}'

        .format(*tmp_nn_output))
pd_results = pd.DataFrame(nn_output,

    columns = ['epoch','total_epochs','train_loss','train_acc','valid_loss','valid_acc','test_loss','test_acc'])

                         

display(pd_results)



print("Best Epoch: {}".format(pd_results.loc[pd_results.valid_acc.idxmax()]['epoch']))
%matplotlib inline

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].plot(pd_results['epoch'],pd_results['valid_loss'], label='validation_loss')

axes[0].plot(pd_results['epoch'],pd_results['train_loss'], label='train_loss')

# axes[0].plot(pd_results['epoch'],pd_results['test_loss'], label='test_loss')

axes[0].legend()



axes[1].plot(pd_results['epoch'],pd_results['valid_acc'], label='validation_acc')

axes[1].plot(pd_results['epoch'],pd_results['train_acc'], label='train_acc')

# axes[1].plot(pd_results['epoch'],pd_results['test_acc'], label='test_acc')

axes[1].legend()
num_classes = len(classes)



# Use the validation set to make a confusion matrix

net.eval() # good habit I suppose

predictions = torch.LongTensor().to(device) # Tensor for all predictions



# Goes through the test set

for images, _ in val_loader:

    images = images.to(device)

    preds = net(images)

    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)



# Make the confusion matrix

cmt = torch.zeros(num_classes, num_classes, dtype=torch.int32)

for i in range(len(val_labels)):

    cmt[val_labels[i], predictions[i]] += 1
cmt
num_classes = len(classes)



# Use the validation set to make a confusion matrix

net.eval() # good habit I suppose

predictions = torch.LongTensor().to(device) # Tensor for all predictions



# Goes through the test set

for images, _ in test_loader:

    images = images.to(device)

    preds = net(images)

    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)



# Make the confusion matrix

cmt = torch.zeros(num_classes, num_classes, dtype=torch.int32)

for i in range(len(test_labels)):

    cmt[test_labels[i], predictions[i]] += 1
cmt
# Time to get the network's predictions on the test set

# Put the test set in a DataLoader



net.eval() # Safety first

predictions = torch.LongTensor().to(device) # Tensor for all predictions



# Go through the test set, saving the predictions in... 'predictions'

for images in submission_loader:

    images = images.to(device)

    preds = net(images)

    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)
# Read in the sample submission

submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")



# Change the label column to our predictions 

# Have to make sure the predictions Tensor is on the cpu

submission['label'] = predictions.cpu().numpy()

# Write the dataframe to a new csv, not including the index

submission.to_csv("submission.csv", index=False)
submission.head()