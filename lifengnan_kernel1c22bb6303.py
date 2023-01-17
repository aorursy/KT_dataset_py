# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

'''

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))

sub = pd.read_csv('../input/submit_csv.csv')

sub.to_csv('submission.csv',index=False)

# Any results you write to the current directory are saved as output.

'''

import os

import argparse

import time

from datetime import timedelta

import shutil

from sklearn import metrics

from PIL import Image, ImageOps

import torch.utils.data as data

import pandas as pd

from tqdm import tqdm

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import torch.backends.cudnn as cudnn

from tqdm import tqdm

import numpy as np

import torchvision.transforms as transforms

import csv

import torch.nn.functional as F

import math

from functools import partial
import torchvision.transforms as transforms

import random

import time

from datetime import timedelta

from PIL import Image, ImageOps

import numpy as np

import numbers

import math

import torch

import cv2



class GroupScale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):

        self.worker = transforms.Resize(size, interpolation)

    def __call__(self, img_group):

        #for img in img_group:

        #    img=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        #    cv2.imwrite('cv_out.png', img)

        return [self.worker(img) for img in img_group]



class Stack(object):

    def __init__(self, is_flow=False):

        self.is_flow = is_flow

    def __call__(self, img_group):

        #  112 112 3

        return np.stack(img_group, axis=0)

        #1 112 112 3



class GroupCenterCrop(object):

    def __init__(self, size):

        self.worker = transforms.CenterCrop(size)



    def __call__(self, img_group):

        return [self.worker(img) for img in img_group]





class GroupRandomHorizontalFlip(object):

    """Randomly horizontally flips the given PIL.Image with a probability of 0.5

    """

    def __init__(self, is_flow=False):

        self.is_flow = is_flow



    def __call__(self, img_group, is_flow=False):

        v = random.random()

        if v < 0.5:

            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]

            if self.is_flow:

                for i in range(0, len(ret), 2):

                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping

            return ret

        else:

            return img_group



class GroupRandomVerticalFlip(object):

    """Randomly horizontally flips the given PIL.Image with a probability of 0.5

    """

    def __init__(self, is_flow=False):

        self.is_flow = is_flow



    def __call__(self, img_group):

        v = random.random()

        if v < 0.5:

            ret = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_group]

            if self.is_flow:

                for i in range(1, len(ret), 2):

                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping

            return ret

        else:

            return img_group





class GroupNormalize(object):

    def __init__(self, mean, std):

        self.mean = mean

        self.std = std



    def __call__(self, tensor):

        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))

        rep_std = self.std * (tensor.size()[0]//len(self.std))



        # TODO: make efficient

        for t, m, s in zip(tensor, rep_mean, rep_std):

            t.sub_(m).div_(s)



        return tensor







class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):

        self.scales = scales if scales is not None else [1, .875, .75, .66]

        self.max_distort = max_distort

        self.fix_crop = fix_crop

        self.more_fix_crop = more_fix_crop

        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

        self.interpolation = Image.BILINEAR



    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)

        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]

        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]

        return ret_img_group



    def _sample_crop_size(self, im_size):

        image_w, image_h = im_size[0], im_size[1]



        # find a crop size

        base_size = min(image_w, image_h)

        crop_sizes = [int(base_size * x) for x in self.scales]

        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]

        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]



        pairs = []

        for i, h in enumerate(crop_h):

            for j, w in enumerate(crop_w):

                if abs(i - j) <= self.max_distort:

                    pairs.append((w, h))



        crop_pair = random.choice(pairs)

        if not self.fix_crop:

            w_offset = random.randint(0, image_w - crop_pair[0])

            h_offset = random.randint(0, image_h - crop_pair[1])

        else:

            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])



        return crop_pair[0], crop_pair[1], w_offset, h_offset



    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):

        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)

        return random.choice(offsets)



    @staticmethod

    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):

        w_step = (image_w - crop_w) / 4

        h_step = (image_h - crop_h) / 4



        ret = list()

        ret.append((0, 0))  # upper left

        ret.append((4 * w_step, 0))  # upper right

        ret.append((0, 4 * h_step))  # lower left

        ret.append((4 * w_step, 4 * h_step))  # lower right

        ret.append((2 * w_step, 2 * h_step))  # center



        if more_fix_crop:

            ret.append((0, 2 * h_step))  # center left

            ret.append((4 * w_step, 2 * h_step))  # center right

            ret.append((2 * w_step, 4 * h_step))  # lower center

            ret.append((2 * w_step, 0 * h_step))  # upper center



            ret.append((1 * w_step, 1 * h_step))  # upper left quarter

            ret.append((3 * w_step, 1 * h_step))  # upper right quarter

            ret.append((1 * w_step, 3 * h_step))  # lower left quarter

            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter



        return ret



class GroupColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):

        self.worker = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)



    def __call__(self, img_group):

        return [self.worker(img) for img in img_group]



class GroupRandomRotate90(object):

    def __init__(self, is_flow=False):

        self.is_flow = is_flow



    def __call__(self, img_group):

        v = random.random()

        if v < 0.5:

            ret = [img.rotate(90) for img in img_group]

            if self.is_flow:

                ret2 = []

                for i in range(0, len(ret), 2):

                    ret2.append(ret[i+1])

                    ret2.append(ret[i])  #change x, y flow when rotate

                ret = ret2

            return ret

        else:

            return img_group





class ToTorchFormatTensor(object):

    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]

    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):

        self.div = div

    def __call__(self, pic):

        img = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous()

        return img.float().div(255) if self.div else img.float()





class GroupRandomRotateAnyAngle_byLFN(object):

    def __init__(self, angle=10):

        self.angle = angle



    def __call__(self, img_group):

        v = random.uniform(-self.angle, self.angle)

        ret = [img.rotate(v, expand=True) for img in img_group]

        return ret





class LFN_crop(object):

    def __init__(self, out_channel=1):

        self.out_channel = out_channel

    def __call__(self, img_group):

        #start_time = time.time()

        img_group1 = [cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) for img in img_group]

        img_group2 = [LFN1_crop_and_resize(img) for img in img_group1]

        img_group3 = [Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) for img in img_group2]

        #print(str(timedelta(seconds=time.time()-start_time)))

        return img_group3

def LFN1_crop_and_resize(img):

    IMG_SIZE = 512

    image = LFN1_crop_image3(img)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    return image





class LFN_trick_1(object):

    def __init__(self, out_channel=1):

        self.out_channel = out_channel

    def __call__(self, img_group):

        #start_time = time.time()

        img_group1 = [cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) for img in img_group]

        img_group2 = [LFN1_gray_and_crop(img) for img in img_group1]

        img_group3 = [Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) for img in img_group2]

        #print(str(timedelta(seconds=time.time()-start_time)))

        return img_group3

def LFN1_gray_and_crop(img):

    dpi = 80 #inch

    IMG_SIZE = 512

    image = LFN1_crop_image3(img)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted(image,4, cv2.GaussianBlur(image,(0,0),IMG_SIZE/10),-4,128)

    return image

def LFN1_crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance  

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]

# The above code work only for 1-channel. Here is my simple extension for 3-channels image

def LFN1_crop_image3(img,tol=7):

    h,w,_=img.shape

    img1=cv2.resize(LFN1_crop_image1(img[:,:,0]),(w,h))

    img2=cv2.resize(LFN1_crop_image1(img[:,:,1]),(w,h))

    img3=cv2.resize(LFN1_crop_image1(img[:,:,2]),(w,h))

    img[:,:,0]=img1

    img[:,:,1]=img2

    img[:,:,2]=img3

    return img



Transforms = {

                'val':transforms.Compose([

                                            LFN_trick_1(),

                                            Stack(is_flow=False),

                                            ToTorchFormatTensor(div=True),

                                            GroupNormalize(mean=[110.63666788 / 255., 103.16065604 / 255., 96.29023126 / 255.], std=[38.7568578 / 255., 37.88248729 / 255., 40.02898126 / 255.])

                                            ])

            }
def conv3x3(in_planes, out_planes, stride=1):

    # 3x3 convolution with padding

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)





def downsample_basic_block(x, planes, stride):

    out = F.avg_pool2d(x, kernel_size=1, stride=stride)

    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3),out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatTensor):

        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out





class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:

            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out



class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        residual = x

        #print('^^^^^^^^^^^^^1',x.shape)

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        #print('^^^^^^^^^^^^^2',out.shape)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        #print('^^^^^^^^^^^^^3',out.shape)

        out = self.conv3(out)

        out = self.bn3(out)#256

        #print('^^^^^^^^^^^^^4',out.shape)

        if self.downsample is not None:

            residual = self.downsample(x)

            #print('highway')

        #print('%%%%%%%%%%%%%%%',residual.shape)

        out += residual

        #print('###############',out.shape)

        out = self.relu(out)



        return out



class ResNet(nn.Module):



    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type, num_classes, dropout):

        self.inplanes = 64



        super(ResNet, self).__init__()



        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)



        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)

        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)



        last_duration = int(math.ceil(sample_duration / 16.))

        last_size = int(math.ceil(sample_size / 32.))



        self.avgpool = nn.AvgPool2d((last_size, last_size), stride=1)

        self.drop = nn.Dropout(p=dropout)

        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):

        downsample = None

        

        if stride != 1 or self.inplanes != planes * block.expansion: 

        #如果： 步长 != 1 // 输入通道数 != 此块原始输入通道数*此块预计通道扩充倍数

            if shortcut_type == 'A':

                downsample = partial(downsample_basic_block,planes=planes * block.expansion,stride=stride)

            else:

                downsample = nn.Sequential(

                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), 

                    nn.BatchNorm2d(planes * block.expansion))



        layers = []

                               #1024          512       2       No.2

        layers.append(block(self.inplanes, planes, stride, downsample))

        #print('0',self.inplanes, planes)

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes))

            #print(i,self.inplanes, planes)



        return nn.Sequential(*layers)



    def forward(self, x):

        #print('0',x.shape)

        x = self.conv1(x)

        #print('1',x.shape)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        #print('2',x.shape)

        x = self.layer1(x)

        #print('3',x.shape)

        x = self.layer2(x)

        #print('4',x.shape)

        x = self.layer3(x)

        #print('5',x.shape)

        x = self.layer4(x)

        #print('6',x.shape)





        x = self.avgpool(x)

        #print('7',x.shape)



        x = x.view(x.size(0), -1)

        x = self.drop(x)

        x = self.fc(x)



        return x



def resnet34(**kwargs):

    """Constructs a ResNet-101 model.

    """

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model



def resnet50(**kwargs):

    """Constructs a ResNet-101 model.

    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model



def resnet101(**kwargs):

    """Constructs a ResNet-101 model.

    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model



def resnet152(**kwargs):

    """Constructs a ResNet-101 model.

    """

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



model_dict = {

'34': [resnet34, 'A'],

'50': [resnet50, 'B'],

'101':[resnet101, 'B'],

'152':[resnet152, 'B']

}

weights = '../input/weights2/blind_resnet50_best_80.08298754131481.pth'

model = model_dict['50'][0](sample_size=512, sample_duration=16, shortcut_type=model_dict['50'], num_classes=5, dropout=0)

model = torch.nn.DataParallel(model)

model = model.cuda()

ckpt = torch.load(weights)

model.load_state_dict(ckpt['state_dict'])

model.eval()

print('Validating...')
imgs_dict = {}

final = []

imgs = pd.read_csv(os.path.join('../input/aptos2019-blindness-detection', 'test.csv'), usecols=[0])

img_group = []

num = imgs.shape[0]

sub = []



for i in tqdm(range(num)):

    final.append(imgs.loc[i][0])

    img_name = final[i]+'.png'

    #print(img_name)

    transform_val = Transforms['val']

    img_group.append(Image.open('../input/aptos2019-blindness-detection/test_images/' + img_name))

    img = transform_val(img_group)

    img_group = []

    img = img.view(-1,3,512,512)

    #print(img_name,img.shape)

    input_ = img

    input_var = Variable(input_)

    with torch.no_grad():

        output = model(input_var)



    _, pred = output.topk(1, 1, True, True)

    pred_1 = pred.t()

    #print(pred_1.item())

    sub.append(str(pred_1.item()))

    #print(i,sub[i])

print(len(sub))

sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

#sample.diagnosis = sub

sample['diagnosis'] = sub

sample.to_csv("submission.csv", index=False)

sample.head()

print ("write over")
