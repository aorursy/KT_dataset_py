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
!pip install geffnet
!pip install git+https://github.com/pabloppp/pytorch-tools -U

import torch
from fastai.vision import *
import torch.utils.data as Data
import cv2
import albumentations as A
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import geffnet
from collections import defaultdict
from torchtools.optim import RangerLars
train_df=pd.read_csv('/kaggle/input/futurefish/training.csv')
train_df.head(20)
label_csv=pd.read_csv('/kaggle/input/futurefish/species.csv')
label_csv.head()
test_csv=pd.read_csv('/kaggle/input/futurefish/annotation.csv')
test_csv.head(20)


import torchvision.transforms as transforms
from PIL import Image as I
class TrainDataset(Data.Dataset):
    def __init__(self,names,image_labels,labels_map_images,transform):
        super(TrainDataset,self).__init__()
        self.names=names
        self.image_labels=image_labels
        self.transform=transform
    
        self.num_class=20
        self.labels_map_images=labels_map_images
        
    def read_image(self,name):
        image=I.open(name)
        image=self.transform(image)
        return image
        
    def __getitem__(self,index):
        
        name=self.names[index]
       
        if type(name)==list:
            
            name=name[0]
        label=self.image_labels[name]
        negative_label=np.random.choice(list(set(list(range(self.num_class)))^set([label])))
        
        negative_name=np.random.choice(self.labels_map_images[negative_label])
        positive_name=np.random.choice(list(set(self.labels_map_images[label])^set([name])))
        
        
        image=self.read_image(name)
        
        positive_image=self.read_image(positive_name)
        
        negative_image=self.read_image(negative_name)
        return [image,positive_image,negative_image],[label,label,negative_label]
        
    def __len__(self):
        return len(self.names)
    
class TestDataset(Data.Dataset):
    def __init__(self,names,image_labels,transform):
        super(TestDataset,self).__init__()
        self.names=names
        self.image_labels=image_labels
        self.transform=transform
        
    def __getitem__(self,index):
        name=self.names[index]
        if type(name)==list:
            name=name[0]
        label=self.image_labels[name]
        image=I.open(name)
        image=self.transform(image)
        return image,label
        
    def __len__(self):
        return len(self.names)

def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels

class Residual(nn.Module):
    def __init__(self, in_channel, R=8, k=2):
        super(Residual, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.R = R
        self.k = k
        out_channel = int(in_channel / R)
        self.fc1 = nn.Linear(in_channel, out_channel)
        fc_list = []
        for i in range(k):
            fc_list.append(nn.Linear(out_channel, 2 * in_channel))
        self.fc2 = nn.ModuleList(fc_list)

    def forward(self, x):
        x = self.avg(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.relu(x)
        result_list = []
        for i in range(self.k):
            result = self.fc2[i](x)
            result = 2 * torch.sigmoid(result) - 1
            result_list.append(result)
        return result_list


class Dynamic_relu_b(nn.Module):
    def __init__(self, inchannel, R=8, k=2):
        super(Dynamic_relu_b, self).__init__()
        self.lambda_alpha = 1
        self.lambda_beta = 0.5
        self.R = R
        self.k = k
        self.init_alpha = torch.zeros(self.k)
        self.init_beta = torch.zeros(self.k)
        self.init_alpha[0] = 1
        self.init_beta[0] = 1
        for i in range(1, k):
            self.init_alpha[i] = 0
            self.init_beta[i] = 0

        self.residual = Residual(inchannel)

    def forward(self, input):
        delta = self.residual(input)
        in_channel = input.shape[1]
        bs = input.shape[0]
        alpha = torch.zeros((self.k, bs, in_channel),device=input.device)
        beta = torch.zeros((self.k, bs, in_channel),device=input.device)
        for i in range(self.k):
            for j, c in enumerate(range(0, in_channel * 2, 2)):
                alpha[i, :, j] = delta[i][:, c]
                beta[i, :, j] = delta[i][:, c + 1]
        alpha1 = alpha[0]
        beta1 = beta[0]
        max_result = self.dynamic_function(alpha1, beta1, input, 0)
        for i in range(1, self.k):
            alphai = alpha[i]
            betai = beta[i]
            result = self.dynamic_function(alphai, betai, input, i)
            max_result = torch.max(max_result, result)
        return max_result
    def dynamic_function(self, alpha, beta, x, k):
        init_alpha = self.init_alpha[k]
        init_alpha=init_alpha.to(x.device)
        init_beta = self.init_beta[k]
        init_beta=init_beta.to(x.device)
        # lambda_alpha=self.lambda_alpha.to(x.device)
        # lambda_beta=self.lambda_beta.to(x.device)
        alpha = init_alpha +  self.lambda_alpha* alpha
        beta = init_beta + self.lambda_beta * beta
        bs = x.shape[0]
        channel = x.shape[1]
        results = torch.zeros_like(x,device=x.device)
        results = x * alpha.view(bs, channel, 1, 1) + beta.view(bs, channel, 1, 1)
        return results


class k_max_pool(nn.Module):
    def __init__(self,k=4):
        super(k_max_pool,self).__init__()
        self.k=k
        self.pool=nn.AdaptiveAvgPool1d(1)
        
    def forward(self,x):
        x=x.view(x.shape[0],x.shape[1],-1)
        x=x.topk(self.k,dim=-1).values
        x=self.pool(x)
        return x
    
    
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

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
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
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

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.conv2 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=4, stride=1,bias=False)
        # self.conv3 = nn.Conv2d(in_channels=2048, out_channels=201, kernel_size=1, stride=1, bias=False)
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
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.act=Dynamic_relu_b(64)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        conv5_b = self.layer4[:2](x)
        x = self.layer4[2](conv5_b)

        fm = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        embeeding = x


        return fm, embeeding, conv5_b
from collections import OrderedDict


def _resnet(arch, block, layers, pretrained, pth_path, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        
        state_dict = torch.load(pth_path)
        model_dict=model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    
    return model

def resnext50_32x4d(pth_path, pretrained=False, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, pth_path, **kwargs)


def resnet50(pth_path, pretrained=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, pth_path,
                   **kwargs)



import torch
from skimage import measure

def AOLM(fms, fm1):
    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float()


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))


        intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox


        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


# def _bn_function_factory(norm, relu, conv):
#     def bn_function(*inputs):
#         concated_features = torch.cat(inputs, 1)
#         bottleneck_output = conv(relu(norm(concated_features)))
#         return bottleneck_output

#     return bn_function


# class _DenseLayer(nn.Module):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
#         super(_DenseLayer, self).__init__()
#         self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
#         self.add_module('relu1', nn.ReLU(inplace=True)),
#         self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
#                         kernel_size=1, stride=1, bias=False)),
#         self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
#         self.add_module('relu2', nn.ReLU(inplace=True)),
#         self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
#                         kernel_size=3, stride=1, padding=1, bias=False)),
#         self.drop_rate = drop_rate
#         self.efficient = efficient

#     def forward(self, *prev_features):
#         bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
#         if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
#             bottleneck_output = cp.checkpoint(bn_function, *prev_features)
#         else:
#             bottleneck_output = bn_function(*prev_features)
#         new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         return new_features


# class _Transition(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features):
#         super(_Transition, self).__init__()
#         self.add_module('norm', nn.BatchNorm2d(num_input_features))
#         self.add_module('relu', nn.ReLU(inplace=True))
#         self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
#                                           kernel_size=1, stride=1, bias=False))
#         self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# class _DenseBlock(nn.Module):
#     def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
#         super(_DenseBlock, self).__init__()
#         for i in range(num_layers):
#             layer = _DenseLayer(
#                 num_input_features + i * growth_rate,
#                 growth_rate=growth_rate,
#                 bn_size=bn_size,
#                 drop_rate=drop_rate,
#                 efficient=efficient,
#             )
#             self.add_module('denselayer%d' % (i + 1), layer)

#     def forward(self, init_features):
#         features = [init_features]
#         for name, layer in self.named_children():
#             new_features = layer(*features)
#             features.append(new_features)
#         return torch.cat(features, 1)


# class DenseNet(nn.Module):
#     r"""Densenet-BC model class, based on
#     `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
#     Args:
#         growth_rate (int) - how many filters to add each layer (`k` in paper)
#         block_config (list of 3 or 4 ints) - how many layers in each pooling block
#         num_init_features (int) - the number of filters to learn in the first convolution layer
#         bn_size (int) - multiplicative factor for number of bottle neck layers
#             (i.e. bn_size * k features in the bottleneck layer)
#         drop_rate (float) - dropout rate after each dense layer
#         num_classes (int) - number of classification classes
#         small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
#         efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
#     """
#     def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
#                  num_init_features=24, bn_size=4, drop_rate=0,
#                  num_classes=10, small_inputs=False, efficient=True):

#         super(DenseNet, self).__init__()
#         assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

#         # First convolution
#         if small_inputs:
#             self.features = nn.Sequential(OrderedDict([
#                 ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
#             ]))
#         else:
#             self.features = nn.Sequential(OrderedDict([
#                 ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
#             ]))
#             self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
#             self.features.add_module('relu0', nn.ReLU(inplace=True))
#             self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
#                                                            ceil_mode=False))

#         # Each denseblock
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=num_features,
#                 bn_size=bn_size,
#                 growth_rate=growth_rate,
#                 drop_rate=drop_rate,
#                 efficient=efficient,
#             )
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features,
#                                     num_output_features=int(num_features * compression))
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = int(num_features * compression)

#         # Final batch norm
#         self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

#         # Linear layer
#         self.classifier = nn.Linear(num_features, num_classes)
        
#         self.act=Dynamic_relu_b(342)

#         # Initialization
#         for name, param in self.named_parameters():
#             if 'conv' in name and 'weight' in name:
#                 n = param.size(0) * param.size(2) * param.size(3)
#                 param.data.normal_().mul_(math.sqrt(2. / n))
#             elif 'norm' in name and 'weight' in name:
#                 param.data.fill_(1)
#             elif 'norm' in name and 'bias' in name:
#                 param.data.fill_(0)
#             elif 'classifier' in name and 'bias' in name:
#                 param.data.fill_(0)

#     def forward(self, x):
#         features = self.features(x)
#         out = self.act(features)
# #         out = F.adaptive_avg_pool2d(out, (1, 1))
# #         out = torch.flatten(out, 1)
# #         out = self.classifier(out)
#         return out

import torch
import torch.nn as nn



#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class DenseBottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution 
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and 
        #we refer to our network with such a bottleneck layer, i.e., 
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` , 
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments 
        #consist of a batch normalization layer and an 1×1 
        #convolutional layer followed by a 2×2 average pooling 
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution 
        #with 16 (or twice the growth rate for DenseNet-BC) 
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each 
        #side of the inputs is zero-padded by one pixel to keep 
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False) 

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the 
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression 
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        
        self.features.add_module('relu',Dynamic_relu_b(inner_channels))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(DenseBottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
    return DenseNet(DenseBottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
    return DenseNet(DenseBottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
    return DenseNet(DenseBottleneck, [6,12,36,24], growth_rate=48)
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        backbone = geffnet.efficientnet_b3(pretrained=True)
        act1=Dynamic_relu_b(40)
        act2=Dynamic_relu_b(1536)
#         self.rawcls_net=nn.Linear(2048,20)
#         self.location_module=resnext50_32x4d('/kaggle/input/pretrained-pytorch/resnext50_32x4d-7cdf4587.pth',pretrained=True)
        self.backbone = torch.nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            act1,
            backbone.blocks,
            backbone.conv_head,
            backbone.bn2,
            act2)
#     
     
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)
#         self.global_maxpool = k_max_pool()
        self.global_bn = nn.BatchNorm1d(1536)
        self.global_bn.bias.requires_grad = False
        self.local_conv = nn.Conv2d(1536, 512, 1)
        self.local_bn = nn.BatchNorm2d(512)
        self.local_bn.bias.requires_grad = False
        self.fc = nn.Linear(1536, 20)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
#         fm, embedding, conv5_b = self.location_module(x)
#         raw_logits = self.rawcls_net(embedding)
#         coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))
#         local_imgs = torch.zeros([x.shape[0], 3, 448, 448],device=x.device)
#         for i in range(x.shape[0]):
#             [x0, y0, x1, y1] = coordinates[i]
#             local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(448, 448),
#                                                 mode='bilinear', align_corners=True)
            
#         del coordinates

#         x = self.backbone(local_imgs.detach())
#         del local_imgs
        x=self.backbone(x)
        global_feat = self.global_avgpool(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.global_bn(global_feat)
        global_feat = l2_norm(global_feat)

        local_feat = torch.mean(x, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)

        out = self.fc(global_feat) * 16
        return global_feat, local_feat, out,raw_logits
import torch.nn as nn
import torch
class TripletLoss(nn.Module):

    def __init__(self,margin=0.3):
        super(TripletLoss,self).__init__()
        self.margin=margin
        self.ranking_loss=nn.MarginRankingLoss(margin=margin)

    def shortest_dist(self,dist_mat):
        m, n = dist_mat.size()[:2]
        dist = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if (i == 0) and (j == 0):
                    dist[i][j] = dist_mat[i, j]
                elif (i == 0) and (j > 0):
                    dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
                elif (i > 0) and (j == 0):
                    dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
                else:
                    dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
        dist = dist[-1][-1]
        return dist

    '''局部特征的距离矩阵'''
    def compute_local_dist(self,x,y):
        M,m,d=x.size()
        N,n,d=y.size()
        x = x.contiguous().view(M * m, d)
        y = y.contiguous().view(N * n, d)
        dist_mat = self.comput_dist(x, y)
        dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
        dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
        dist_mat = self.shortest_dist(dist_mat)
        return dist_mat


    '''全局特征的距离矩阵'''
    def comput_dist(self,x,y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def hard_example_mining(self,dist_mat,labels,return_inds=False):
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)


        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        list_ap = []
        list_an = []
        for i in range(N):
            list_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
            list_an.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))
        dist_ap = torch.cat(list_ap)
        dist_an = torch.cat(list_an)
        return dist_ap, dist_an


    def forward(self,feat_type,feat,labels):
        '''

        :param feat_type: 'global'代表计算全局特征的三重损失，'local'代表计算局部特征
        :param feat: 经过网络计算出来的结果
        :param labels: 标签
        :return:
        '''
        if feat_type=='global':
            dist_mat = self.comput_dist(feat,feat)
        else:
            dist_mat=self.compute_local_dist(feat,feat)
        dist_ap, dist_an = self.hard_example_mining(
            dist_mat, labels)
        y=torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
    
def one_hot_smooth_label(x,num_class,smooth=0.1):
    num=x.shape[0]
    labels=torch.zeros((num,20))
    for i in range(num):
        labels[i][x[i]]=1
    labels=(1-(num_class-1)/num_class*smooth)*labels+smooth/num_class
    return labels
class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
            super(CenterLoss,self).__init__()
            self.num_classes=num_classes
            self.feat_dim=feat_dim
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            
    def forward(self, x, labels):
        '''

        :param x: [batch_size,feat_dim]
        :param labels: [batch_size]
        :return:
        '''
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        classes = torch.arange(self.num_classes).long()
        classes=classes
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
  
    
class Criterion:
    def __init__(self):
        self.triplet_criterion=TripletLoss()
        self.cls_criterion=nn.BCEWithLogitsLoss()
        self.center_criterion=CenterLoss(20,342)
    
    def __call__(self,global_feat,local_feat,cls_score,label):
   
        global_loss=self.triplet_criterion('global',global_feat,label)
        center_loss=self.center_criterion(global_feat,label)
        local_loss=self.triplet_criterion('local',local_feat,label)
        label=one_hot_smooth_label(label,20)
        cls_loss=self.cls_criterion(cls_score,label)
#         raw_loss=self.cls_criterion(raw_logits,label)
       
        return global_loss+local_loss+cls_loss+0.0005*center_loss
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path,patience=4, best_score=None, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path=checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_metric, model):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        state = {'best_metric': metric, 'state': model.state_dict()}
        torch.save(state, self.checkpoint_path)
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path,patience=4, best_score=None, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path=checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_metric, model):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        state = {'best_metric': metric, 'state': model.state_dict()}
        torch.save(state, self.checkpoint_path)

from fastai.vision import *
from fastai import *


def evaluate(model,valid_dl):
    steps=len(valid_dl)
    device=torch.device('cuda:0')
    step_loss=0
    step_metric=0
    with torch.no_grad():
        for images,labels in valid_dl:
            images=images.to(device).float()
            global_feat,local_feat,cls_score=model(images)
            cls_score=cls_score.to('cpu')
            metric=accuracy(cls_score,labels)
            step_metric+=metric
    metric=step_metric/steps
    valid_loss=step_loss/steps
    print('metric:{}'.format(metric))
    return metric
from tqdm import tqdm
import pickle

class flat_and_anneal(nn.Module):
    def __init__(self ,epochs ,anneal_start=0.5,base_lr=0.001 ,min_lr=0):
        super(flat_and_anneal ,self).__init__()
        self.epochs =epochs
        self.anneal_start =anneal_start
        self.base_lr =base_lr
        self.min_lr =min_lr


    def forward(self ,epoch ,optimizer):
            if epoch>=15:
                epoch =epoch -15
                for param in optimizer.param_groups:
                    lr =self.min_lr +(self.base_lr -self.min_lr ) *( 1 +math.cos(math.pi *epoch /5) ) /2
                    param['lr' ] =lr
def main(train_dl, valid_dl, k):
  
    criterion = Criterion()
    checkpoint_path = 'step{}.pt'.format(k)
    early_stop = EarlyStopping(checkpoint_path)

    model = myNet()
    device = torch.device('cuda:0')
    model = model.to(device)
    
    epochs=50
    optimizer=RangerLars(model.parameters(),lr=0.001)
    scheduler=flat_and_anneal(epochs)
    
    for epoch in range(epochs):
        with tqdm(total=len(train_dl)) as pbar:
            train_loss = 0
            steps = len(train_dl)
            for image, labels in train_dl:
                model.train()
                optimizer.zero_grad()
              
                image = image.to(device).float()
                global_feat, local_feat, cls_score = model(image)
                global_feat = global_feat.to('cpu')
                cls_score = cls_score.to('cpu')
                local_feat = local_feat.to('cpu')
#                 raw_logits=raw_logits.to('cpu')
                loss = criterion(global_feat, local_feat, cls_score,labels)
                train_loss += loss
                loss.backward()
                optimizer.step()
                pbar.update(1)
            print('epoch:{},train_loss:{}'.format(epoch,train_loss / steps))
            model.eval()
            metric = evaluate(model, valid_dl)
            early_stop(metric, model)
            scheduler(epoch,optimizer)
        
#     step1_optimizer = torch.optim.SGD(model.parameters(), lr=0.9, weight_decay=0.0001)
#     for epoch in range(step1_epochs):
#         with tqdm(total=len(train_dl)) as pbar:
#             train_loss = 0
#             steps = len(train_dl)
#             for image, labels in train_dl:
#                 model.train()
#                 step1_optimizer.zero_grad()
              
#                 image = image.to(device).float()
#                 global_feat, local_feat, cls_score = model(image)
#                 global_feat = global_feat.to('cpu')
#                 cls_score = cls_score.to('cpu')
#                 local_feat = local_feat.to('cpu')
#                 loss = criterion(global_feat, local_feat, cls_score, labels)
#                 train_loss += loss
#                 loss.backward()
#                 step1_optimizer.step()
#                 pbar.update(1)
#             print('train_loss:{}'.format(train_loss / steps))
#             model.eval()
#             metric = evaluate(model, valid_dl)
#             early_stop(metric, model)
#             if early_stop.early_stop:
#                 break

#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['state'])
#     step2_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(step2_optimizer, T_0=5, T_mult=2)
#     early_stop.counter = 0
#     early_stop.early_stop = False
#     early_stop.best_score = 0
#     early_stop.patience = 8
#     for epoch in range(step2_epochs):
#         with tqdm(total=len(train_dl)) as pbar:
#             train_loss = 0
#             steps = len(train_dl)
#             for image, labels in train_dl:
#                 model.train()
#                 step2_optimizer.zero_grad()
              
#                 image = image.to(device).float()
#                 global_feat, local_feat, cls_score = model(image)
#                 global_feat = global_feat.to('cpu')
#                 local_feat = local_feat.to('cpu')
#                 cls_score = cls_score.to('cpu')
#                 loss = criterion(global_feat, local_feat, cls_score, labels)
#                 train_loss += loss
#                 loss.backward()
#                 step2_optimizer.step()
#                 pbar.update(1)
#             print('train_loss:{}'.format(train_loss / steps))
#             model.eval()
#             metric = evaluate(model, valid_dl)
#             scheduler.step()
#             early_stop(metric, model)
#             if early_stop.early_stop:
#                 break


image_dir='/kaggle/input/futurefish/data/data'
train_names=[]
image_labels={}
valid_names=[]
label_map_image=defaultdict(list)
label_set=[]
for i in range(train_df.shape[0]):
    info=train_df.iloc[i]
    name=info['FileID']
    name=os.path.join(image_dir,name+'.jpg')
    train_names.append(name)
    label=info['SpeciesID']
    image_labels[name]=label
    label_map_image[label].append(name)
for i in range(test_csv.shape[0]):
    info=test_csv.iloc[i]
    name=info['FileID']
    name=os.path.join(image_dir,name+'.jpg')
    valid_names.append(name)
    label=info['SpeciesID']
    image_labels[name]=label
    label_map_image[label].append(name)
label_set=list(label_csv['ScientificName'].unique())
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from PIL import ImageOps, ImageEnhance
from PIL import Image as I

class BaseTransform(ABC):

    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
                (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass


class ShearXY(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomAffine(0, shear=degrees, resample=I.BILINEAR)
        return t(img)


class TranslateXY(BaseTransform):

    def transform(self, img):
        translate = (self.mag, self.mag)
        t = transforms.RandomAffine(0, translate=translate, resample=I.BILINEAR)
        return t(img)


class Rotate(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomRotation(degrees, I.BILINEAR)
        return t(img)


class AutoContrast(BaseTransform):

    def transform(self, img):
        cutoff = int(self.mag * 49)
        return ImageOps.autocontrast(img, cutoff=cutoff)


class Invert(BaseTransform):

    def transform(self, img):
        return ImageOps.invert(img)


class Equalize(BaseTransform):

    def transform(self, img):
        return ImageOps.equalize(img)


class Solarize(BaseTransform):

    def transform(self, img):
        threshold = (1-self.mag) * 255
        return ImageOps.solarize(img, threshold)


class Posterize(BaseTransform):

    def transform(self, img):
        bits = int((1-self.mag) * 8)
        return ImageOps.posterize(img, bits=bits)


class Contrast(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Contrast(img).enhance(factor)


class Color(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Color(img).enhance(factor)


class Brightness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Brightness(img).enhance(factor)


class Sharpness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Sharpness(img).enhance(factor)


class Cutout(BaseTransform):

    def transform(self, img):
        n_holes = 1
        length = 24 * self.mag
        cutout_op = CutoutOp(n_holes=n_holes, length=length)
        return cutout_op(img)


class CutoutOp(object):
    """
    https://github.com/uoguelph-mlrg/Cutout
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        w, h = img.size

        mask = np.ones((h, w, 1), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h).astype(int)
            y2 = np.clip(y + self.length // 2, 0, h).astype(int)
            x1 = np.clip(x - self.length // 2, 0, w).astype(int)
            x2 = np.clip(x + self.length // 2, 0, w).astype(int)

            mask[y1: y2, x1: x2, :] = 0.

        img = mask*np.asarray(img).astype(np.uint8)
        img = I.fromarray(mask*np.asarray(img))

        return img
transforms_train=[]
for i in range(5):
    with open('/kaggle/input/linshi/transform_list{}.txt'.format(i),'rb') as f:
            transform=pickle.load(f)
    transforms_train.extend(transform)                 

transform_train=transforms.Compose([
        transforms.Resize([448,448],I.BILINEAR),

        transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(45),
#         transforms.ColorJitter(),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         transforms.RandomErasing()        ]


transform_valid=transforms.Compose([
    transforms.Resize([448,448],I.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print(transform_train)
k=0

train_label_map_image=defaultdict(list)
for name in train_names:
    name=name
    label=image_labels[name]
    train_label_map_image[label].append(name)
train_ds=TrainDataset(train_names,image_labels,train_label_map_image,transform_train)
valid_ds=TestDataset(valid_names,image_labels,transform_valid)
train_dl=Data.DataLoader(train_ds,batch_size=8,collate_fn=train_collate,shuffle=True,drop_last=True)
valid_dl=Data.DataLoader(valid_ds,batch_size=8,drop_last=True)
main(train_dl,valid_dl,k)
