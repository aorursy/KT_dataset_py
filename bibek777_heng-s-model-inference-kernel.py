!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null

#package_path = '../input/unet-model'

package_path = '../input/unetmodelscript' # add unet script dataset

import sys

sys.path.append(package_path)
# Get necessary Imports

import pdb

import os

import cv2

import torch

import pandas as pd

import numpy as np

from tqdm import tqdm

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Dataset

from albumentations import (Normalize, Compose)

from albumentations.torch import ToTensor

import torch.utils.data as data

import torchvision.models as models

import torch.nn as nn

from torch.nn import functional as F

from model import Unet
# Codes from Heng's baseline

# This code is for classifcation model



BatchNorm2d = nn.BatchNorm2d



IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]

IMAGE_RGB_STD  = [0.229, 0.224, 0.225]





###############################################################################

CONVERSION=[

 'block0.0.weight',	(64, 3, 7, 7),	 'conv1.weight',	(64, 3, 7, 7),

 'block0.1.weight',	(64,),	 'bn1.weight',	(64,),

 'block0.1.bias',	(64,),	 'bn1.bias',	(64,),

 'block0.1.running_mean',	(64,),	 'bn1.running_mean',	(64,),

 'block0.1.running_var',	(64,),	 'bn1.running_var',	(64,),

 'block1.1.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv1.weight',	(64, 64, 3, 3),

 'block1.1.conv_bn1.bn.weight',	(64,),	 'layer1.0.bn1.weight',	(64,),

 'block1.1.conv_bn1.bn.bias',	(64,),	 'layer1.0.bn1.bias',	(64,),

 'block1.1.conv_bn1.bn.running_mean',	(64,),	 'layer1.0.bn1.running_mean',	(64,),

 'block1.1.conv_bn1.bn.running_var',	(64,),	 'layer1.0.bn1.running_var',	(64,),

 'block1.1.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv2.weight',	(64, 64, 3, 3),

 'block1.1.conv_bn2.bn.weight',	(64,),	 'layer1.0.bn2.weight',	(64,),

 'block1.1.conv_bn2.bn.bias',	(64,),	 'layer1.0.bn2.bias',	(64,),

 'block1.1.conv_bn2.bn.running_mean',	(64,),	 'layer1.0.bn2.running_mean',	(64,),

 'block1.1.conv_bn2.bn.running_var',	(64,),	 'layer1.0.bn2.running_var',	(64,),

 'block1.2.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv1.weight',	(64, 64, 3, 3),

 'block1.2.conv_bn1.bn.weight',	(64,),	 'layer1.1.bn1.weight',	(64,),

 'block1.2.conv_bn1.bn.bias',	(64,),	 'layer1.1.bn1.bias',	(64,),

 'block1.2.conv_bn1.bn.running_mean',	(64,),	 'layer1.1.bn1.running_mean',	(64,),

 'block1.2.conv_bn1.bn.running_var',	(64,),	 'layer1.1.bn1.running_var',	(64,),

 'block1.2.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv2.weight',	(64, 64, 3, 3),

 'block1.2.conv_bn2.bn.weight',	(64,),	 'layer1.1.bn2.weight',	(64,),

 'block1.2.conv_bn2.bn.bias',	(64,),	 'layer1.1.bn2.bias',	(64,),

 'block1.2.conv_bn2.bn.running_mean',	(64,),	 'layer1.1.bn2.running_mean',	(64,),

 'block1.2.conv_bn2.bn.running_var',	(64,),	 'layer1.1.bn2.running_var',	(64,),

 'block1.3.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv1.weight',	(64, 64, 3, 3),

 'block1.3.conv_bn1.bn.weight',	(64,),	 'layer1.2.bn1.weight',	(64,),

 'block1.3.conv_bn1.bn.bias',	(64,),	 'layer1.2.bn1.bias',	(64,),

 'block1.3.conv_bn1.bn.running_mean',	(64,),	 'layer1.2.bn1.running_mean',	(64,),

 'block1.3.conv_bn1.bn.running_var',	(64,),	 'layer1.2.bn1.running_var',	(64,),

 'block1.3.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv2.weight',	(64, 64, 3, 3),

 'block1.3.conv_bn2.bn.weight',	(64,),	 'layer1.2.bn2.weight',	(64,),

 'block1.3.conv_bn2.bn.bias',	(64,),	 'layer1.2.bn2.bias',	(64,),

 'block1.3.conv_bn2.bn.running_mean',	(64,),	 'layer1.2.bn2.running_mean',	(64,),

 'block1.3.conv_bn2.bn.running_var',	(64,),	 'layer1.2.bn2.running_var',	(64,),

 'block2.0.conv_bn1.conv.weight',	(128, 64, 3, 3),	 'layer2.0.conv1.weight',	(128, 64, 3, 3),

 'block2.0.conv_bn1.bn.weight',	(128,),	 'layer2.0.bn1.weight',	(128,),

 'block2.0.conv_bn1.bn.bias',	(128,),	 'layer2.0.bn1.bias',	(128,),

 'block2.0.conv_bn1.bn.running_mean',	(128,),	 'layer2.0.bn1.running_mean',	(128,),

 'block2.0.conv_bn1.bn.running_var',	(128,),	 'layer2.0.bn1.running_var',	(128,),

 'block2.0.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.0.conv2.weight',	(128, 128, 3, 3),

 'block2.0.conv_bn2.bn.weight',	(128,),	 'layer2.0.bn2.weight',	(128,),

 'block2.0.conv_bn2.bn.bias',	(128,),	 'layer2.0.bn2.bias',	(128,),

 'block2.0.conv_bn2.bn.running_mean',	(128,),	 'layer2.0.bn2.running_mean',	(128,),

 'block2.0.conv_bn2.bn.running_var',	(128,),	 'layer2.0.bn2.running_var',	(128,),

 'block2.0.shortcut.conv.weight',	(128, 64, 1, 1),	 'layer2.0.downsample.0.weight',	(128, 64, 1, 1),

 'block2.0.shortcut.bn.weight',	(128,),	 'layer2.0.downsample.1.weight',	(128,),

 'block2.0.shortcut.bn.bias',	(128,),	 'layer2.0.downsample.1.bias',	(128,),

 'block2.0.shortcut.bn.running_mean',	(128,),	 'layer2.0.downsample.1.running_mean',	(128,),

 'block2.0.shortcut.bn.running_var',	(128,),	 'layer2.0.downsample.1.running_var',	(128,),

 'block2.1.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv1.weight',	(128, 128, 3, 3),

 'block2.1.conv_bn1.bn.weight',	(128,),	 'layer2.1.bn1.weight',	(128,),

 'block2.1.conv_bn1.bn.bias',	(128,),	 'layer2.1.bn1.bias',	(128,),

 'block2.1.conv_bn1.bn.running_mean',	(128,),	 'layer2.1.bn1.running_mean',	(128,),

 'block2.1.conv_bn1.bn.running_var',	(128,),	 'layer2.1.bn1.running_var',	(128,),

 'block2.1.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv2.weight',	(128, 128, 3, 3),

 'block2.1.conv_bn2.bn.weight',	(128,),	 'layer2.1.bn2.weight',	(128,),

 'block2.1.conv_bn2.bn.bias',	(128,),	 'layer2.1.bn2.bias',	(128,),

 'block2.1.conv_bn2.bn.running_mean',	(128,),	 'layer2.1.bn2.running_mean',	(128,),

 'block2.1.conv_bn2.bn.running_var',	(128,),	 'layer2.1.bn2.running_var',	(128,),

 'block2.2.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv1.weight',	(128, 128, 3, 3),

 'block2.2.conv_bn1.bn.weight',	(128,),	 'layer2.2.bn1.weight',	(128,),

 'block2.2.conv_bn1.bn.bias',	(128,),	 'layer2.2.bn1.bias',	(128,),

 'block2.2.conv_bn1.bn.running_mean',	(128,),	 'layer2.2.bn1.running_mean',	(128,),

 'block2.2.conv_bn1.bn.running_var',	(128,),	 'layer2.2.bn1.running_var',	(128,),

 'block2.2.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv2.weight',	(128, 128, 3, 3),

 'block2.2.conv_bn2.bn.weight',	(128,),	 'layer2.2.bn2.weight',	(128,),

 'block2.2.conv_bn2.bn.bias',	(128,),	 'layer2.2.bn2.bias',	(128,),

 'block2.2.conv_bn2.bn.running_mean',	(128,),	 'layer2.2.bn2.running_mean',	(128,),

 'block2.2.conv_bn2.bn.running_var',	(128,),	 'layer2.2.bn2.running_var',	(128,),

 'block2.3.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv1.weight',	(128, 128, 3, 3),

 'block2.3.conv_bn1.bn.weight',	(128,),	 'layer2.3.bn1.weight',	(128,),

 'block2.3.conv_bn1.bn.bias',	(128,),	 'layer2.3.bn1.bias',	(128,),

 'block2.3.conv_bn1.bn.running_mean',	(128,),	 'layer2.3.bn1.running_mean',	(128,),

 'block2.3.conv_bn1.bn.running_var',	(128,),	 'layer2.3.bn1.running_var',	(128,),

 'block2.3.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv2.weight',	(128, 128, 3, 3),

 'block2.3.conv_bn2.bn.weight',	(128,),	 'layer2.3.bn2.weight',	(128,),

 'block2.3.conv_bn2.bn.bias',	(128,),	 'layer2.3.bn2.bias',	(128,),

 'block2.3.conv_bn2.bn.running_mean',	(128,),	 'layer2.3.bn2.running_mean',	(128,),

 'block2.3.conv_bn2.bn.running_var',	(128,),	 'layer2.3.bn2.running_var',	(128,),

 'block3.0.conv_bn1.conv.weight',	(256, 128, 3, 3),	 'layer3.0.conv1.weight',	(256, 128, 3, 3),

 'block3.0.conv_bn1.bn.weight',	(256,),	 'layer3.0.bn1.weight',	(256,),

 'block3.0.conv_bn1.bn.bias',	(256,),	 'layer3.0.bn1.bias',	(256,),

 'block3.0.conv_bn1.bn.running_mean',	(256,),	 'layer3.0.bn1.running_mean',	(256,),

 'block3.0.conv_bn1.bn.running_var',	(256,),	 'layer3.0.bn1.running_var',	(256,),

 'block3.0.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.0.conv2.weight',	(256, 256, 3, 3),

 'block3.0.conv_bn2.bn.weight',	(256,),	 'layer3.0.bn2.weight',	(256,),

 'block3.0.conv_bn2.bn.bias',	(256,),	 'layer3.0.bn2.bias',	(256,),

 'block3.0.conv_bn2.bn.running_mean',	(256,),	 'layer3.0.bn2.running_mean',	(256,),

 'block3.0.conv_bn2.bn.running_var',	(256,),	 'layer3.0.bn2.running_var',	(256,),

 'block3.0.shortcut.conv.weight',	(256, 128, 1, 1),	 'layer3.0.downsample.0.weight',	(256, 128, 1, 1),

 'block3.0.shortcut.bn.weight',	(256,),	 'layer3.0.downsample.1.weight',	(256,),

 'block3.0.shortcut.bn.bias',	(256,),	 'layer3.0.downsample.1.bias',	(256,),

 'block3.0.shortcut.bn.running_mean',	(256,),	 'layer3.0.downsample.1.running_mean',	(256,),

 'block3.0.shortcut.bn.running_var',	(256,),	 'layer3.0.downsample.1.running_var',	(256,),

 'block3.1.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv1.weight',	(256, 256, 3, 3),

 'block3.1.conv_bn1.bn.weight',	(256,),	 'layer3.1.bn1.weight',	(256,),

 'block3.1.conv_bn1.bn.bias',	(256,),	 'layer3.1.bn1.bias',	(256,),

 'block3.1.conv_bn1.bn.running_mean',	(256,),	 'layer3.1.bn1.running_mean',	(256,),

 'block3.1.conv_bn1.bn.running_var',	(256,),	 'layer3.1.bn1.running_var',	(256,),

 'block3.1.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv2.weight',	(256, 256, 3, 3),

 'block3.1.conv_bn2.bn.weight',	(256,),	 'layer3.1.bn2.weight',	(256,),

 'block3.1.conv_bn2.bn.bias',	(256,),	 'layer3.1.bn2.bias',	(256,),

 'block3.1.conv_bn2.bn.running_mean',	(256,),	 'layer3.1.bn2.running_mean',	(256,),

 'block3.1.conv_bn2.bn.running_var',	(256,),	 'layer3.1.bn2.running_var',	(256,),

 'block3.2.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv1.weight',	(256, 256, 3, 3),

 'block3.2.conv_bn1.bn.weight',	(256,),	 'layer3.2.bn1.weight',	(256,),

 'block3.2.conv_bn1.bn.bias',	(256,),	 'layer3.2.bn1.bias',	(256,),

 'block3.2.conv_bn1.bn.running_mean',	(256,),	 'layer3.2.bn1.running_mean',	(256,),

 'block3.2.conv_bn1.bn.running_var',	(256,),	 'layer3.2.bn1.running_var',	(256,),

 'block3.2.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv2.weight',	(256, 256, 3, 3),

 'block3.2.conv_bn2.bn.weight',	(256,),	 'layer3.2.bn2.weight',	(256,),

 'block3.2.conv_bn2.bn.bias',	(256,),	 'layer3.2.bn2.bias',	(256,),

 'block3.2.conv_bn2.bn.running_mean',	(256,),	 'layer3.2.bn2.running_mean',	(256,),

 'block3.2.conv_bn2.bn.running_var',	(256,),	 'layer3.2.bn2.running_var',	(256,),

 'block3.3.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv1.weight',	(256, 256, 3, 3),

 'block3.3.conv_bn1.bn.weight',	(256,),	 'layer3.3.bn1.weight',	(256,),

 'block3.3.conv_bn1.bn.bias',	(256,),	 'layer3.3.bn1.bias',	(256,),

 'block3.3.conv_bn1.bn.running_mean',	(256,),	 'layer3.3.bn1.running_mean',	(256,),

 'block3.3.conv_bn1.bn.running_var',	(256,),	 'layer3.3.bn1.running_var',	(256,),

 'block3.3.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv2.weight',	(256, 256, 3, 3),

 'block3.3.conv_bn2.bn.weight',	(256,),	 'layer3.3.bn2.weight',	(256,),

 'block3.3.conv_bn2.bn.bias',	(256,),	 'layer3.3.bn2.bias',	(256,),

 'block3.3.conv_bn2.bn.running_mean',	(256,),	 'layer3.3.bn2.running_mean',	(256,),

 'block3.3.conv_bn2.bn.running_var',	(256,),	 'layer3.3.bn2.running_var',	(256,),

 'block3.4.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv1.weight',	(256, 256, 3, 3),

 'block3.4.conv_bn1.bn.weight',	(256,),	 'layer3.4.bn1.weight',	(256,),

 'block3.4.conv_bn1.bn.bias',	(256,),	 'layer3.4.bn1.bias',	(256,),

 'block3.4.conv_bn1.bn.running_mean',	(256,),	 'layer3.4.bn1.running_mean',	(256,),

 'block3.4.conv_bn1.bn.running_var',	(256,),	 'layer3.4.bn1.running_var',	(256,),

 'block3.4.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv2.weight',	(256, 256, 3, 3),

 'block3.4.conv_bn2.bn.weight',	(256,),	 'layer3.4.bn2.weight',	(256,),

 'block3.4.conv_bn2.bn.bias',	(256,),	 'layer3.4.bn2.bias',	(256,),

 'block3.4.conv_bn2.bn.running_mean',	(256,),	 'layer3.4.bn2.running_mean',	(256,),

 'block3.4.conv_bn2.bn.running_var',	(256,),	 'layer3.4.bn2.running_var',	(256,),

 'block3.5.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv1.weight',	(256, 256, 3, 3),

 'block3.5.conv_bn1.bn.weight',	(256,),	 'layer3.5.bn1.weight',	(256,),

 'block3.5.conv_bn1.bn.bias',	(256,),	 'layer3.5.bn1.bias',	(256,),

 'block3.5.conv_bn1.bn.running_mean',	(256,),	 'layer3.5.bn1.running_mean',	(256,),

 'block3.5.conv_bn1.bn.running_var',	(256,),	 'layer3.5.bn1.running_var',	(256,),

 'block3.5.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv2.weight',	(256, 256, 3, 3),

 'block3.5.conv_bn2.bn.weight',	(256,),	 'layer3.5.bn2.weight',	(256,),

 'block3.5.conv_bn2.bn.bias',	(256,),	 'layer3.5.bn2.bias',	(256,),

 'block3.5.conv_bn2.bn.running_mean',	(256,),	 'layer3.5.bn2.running_mean',	(256,),

 'block3.5.conv_bn2.bn.running_var',	(256,),	 'layer3.5.bn2.running_var',	(256,),

 'block4.0.conv_bn1.conv.weight',	(512, 256, 3, 3),	 'layer4.0.conv1.weight',	(512, 256, 3, 3),

 'block4.0.conv_bn1.bn.weight',	(512,),	 'layer4.0.bn1.weight',	(512,),

 'block4.0.conv_bn1.bn.bias',	(512,),	 'layer4.0.bn1.bias',	(512,),

 'block4.0.conv_bn1.bn.running_mean',	(512,),	 'layer4.0.bn1.running_mean',	(512,),

 'block4.0.conv_bn1.bn.running_var',	(512,),	 'layer4.0.bn1.running_var',	(512,),

 'block4.0.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.0.conv2.weight',	(512, 512, 3, 3),

 'block4.0.conv_bn2.bn.weight',	(512,),	 'layer4.0.bn2.weight',	(512,),

 'block4.0.conv_bn2.bn.bias',	(512,),	 'layer4.0.bn2.bias',	(512,),

 'block4.0.conv_bn2.bn.running_mean',	(512,),	 'layer4.0.bn2.running_mean',	(512,),

 'block4.0.conv_bn2.bn.running_var',	(512,),	 'layer4.0.bn2.running_var',	(512,),

 'block4.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer4.0.downsample.0.weight',	(512, 256, 1, 1),

 'block4.0.shortcut.bn.weight',	(512,),	 'layer4.0.downsample.1.weight',	(512,),

 'block4.0.shortcut.bn.bias',	(512,),	 'layer4.0.downsample.1.bias',	(512,),

 'block4.0.shortcut.bn.running_mean',	(512,),	 'layer4.0.downsample.1.running_mean',	(512,),

 'block4.0.shortcut.bn.running_var',	(512,),	 'layer4.0.downsample.1.running_var',	(512,),

 'block4.1.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv1.weight',	(512, 512, 3, 3),

 'block4.1.conv_bn1.bn.weight',	(512,),	 'layer4.1.bn1.weight',	(512,),

 'block4.1.conv_bn1.bn.bias',	(512,),	 'layer4.1.bn1.bias',	(512,),

 'block4.1.conv_bn1.bn.running_mean',	(512,),	 'layer4.1.bn1.running_mean',	(512,),

 'block4.1.conv_bn1.bn.running_var',	(512,),	 'layer4.1.bn1.running_var',	(512,),

 'block4.1.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv2.weight',	(512, 512, 3, 3),

 'block4.1.conv_bn2.bn.weight',	(512,),	 'layer4.1.bn2.weight',	(512,),

 'block4.1.conv_bn2.bn.bias',	(512,),	 'layer4.1.bn2.bias',	(512,),

 'block4.1.conv_bn2.bn.running_mean',	(512,),	 'layer4.1.bn2.running_mean',	(512,),

 'block4.1.conv_bn2.bn.running_var',	(512,),	 'layer4.1.bn2.running_var',	(512,),

 'block4.2.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv1.weight',	(512, 512, 3, 3),

 'block4.2.conv_bn1.bn.weight',	(512,),	 'layer4.2.bn1.weight',	(512,),

 'block4.2.conv_bn1.bn.bias',	(512,),	 'layer4.2.bn1.bias',	(512,),

 'block4.2.conv_bn1.bn.running_mean',	(512,),	 'layer4.2.bn1.running_mean',	(512,),

 'block4.2.conv_bn1.bn.running_var',	(512,),	 'layer4.2.bn1.running_var',	(512,),

 'block4.2.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv2.weight',	(512, 512, 3, 3),

 'block4.2.conv_bn2.bn.weight',	(512,),	 'layer4.2.bn2.weight',	(512,),

 'block4.2.conv_bn2.bn.bias',	(512,),	 'layer4.2.bn2.bias',	(512,),

 'block4.2.conv_bn2.bn.running_mean',	(512,),	 'layer4.2.bn2.running_mean',	(512,),

 'block4.2.conv_bn2.bn.running_var',	(512,),	 'layer4.2.bn2.running_var',	(512,),

 'logit.weight',	(1000, 512),	 'fc.weight',	(1000, 512),

 'logit.bias',	(1000,),	 'fc.bias',	(1000,),



]



###############################################################################

class ConvBn2d(nn.Module):



    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):

        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)



    def forward(self,x):

        x = self.conv(x)

        x = self.bn(x)

        return x









#############  resnext50 pyramid feature net #######################################

# https://github.com/Hsuxu/ResNeXt/blob/master/models.py

# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py

# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py





# bottleneck type C

class BasicBlock(nn.Module):

    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):

        super(BasicBlock, self).__init__()

        self.is_shortcut = is_shortcut



        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=3, padding=1, stride=stride)

        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=3, padding=1, stride=1)



        if is_shortcut:

            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)





    def forward(self, x):

        z = F.relu(self.conv_bn1(x),inplace=True)

        z = self.conv_bn2(z)



        if self.is_shortcut:

            x = self.shortcut(x)



        z += x

        z = F.relu(z,inplace=True)

        return z









class ResNet34(nn.Module):



    def __init__(self, num_class=1000 ):

        super(ResNet34, self).__init__()





        self.block0  = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),

            BatchNorm2d(64),

            nn.ReLU(inplace=True),

        )

        self.block1  = nn.Sequential(

             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),

          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,3)],

        )

        self.block2  = nn.Sequential(

             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),

          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,4)],

        )

        self.block3  = nn.Sequential(

             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),

          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,6)],

        )

        self.block4 = nn.Sequential(

             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),

          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,3)],

        )

        self.logit = nn.Linear(512,num_class)







    def forward(self, x):

        batch_size = len(x)



        x = self.block0(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)

        logit = self.logit(x)

        return logit
class Resnet34_classification(nn.Module):

    def __init__(self,num_class=4):

        super(Resnet34_classification, self).__init__()

        e = ResNet34()

        self.block = nn.ModuleList([

            e.block0,

            e.block1,

            e.block2,

            e.block3,

            e.block4,

        ])

        e = None  #dropped

        self.feature = nn.Conv2d(512,32, kernel_size=1) #dummy conv for dim reduction

        self.logit = nn.Conv2d(32,num_class, kernel_size=1)



    def forward(self, x):

        batch_size,C,H,W = x.shape



        for i in range( len(self.block)):

            x = self.block[i](x)

            #print(i, x.shape)



        x = F.dropout(x,0.5,training=self.training)

        x = F.adaptive_avg_pool2d(x, 1)

        x = self.feature(x)

        logit = self.logit(x)

        return logit

model_classification = Resnet34_classification()

model_classification.load_state_dict(torch.load('../input/clsification/00007500_model.pth', map_location=lambda storage, loc: storage), strict=True)
# Dataset setup

class TestDataset(Dataset):

    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):

        self.root = root

        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

        self.fnames = df['ImageId'].unique().tolist()

        self.num_samples = len(self.fnames)

        self.transform = Compose(

            [

                Normalize(mean=mean, std=std, p=1),

                ToTensor(),

            ]

        )



    def __getitem__(self, idx):

        fname = self.fnames[idx]

        path = os.path.join(self.root, fname)

        image = cv2.imread(path)

        images = self.transform(image=image)["image"]

        return fname, images



    def __len__(self):

        return self.num_samples
sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'

test_data_folder = "../input/severstal-steel-defect-detection/test_images"
# hyperparameters

batch_size = 1



# mean and std

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
# dataloader

testset = DataLoader(

    TestDataset(test_data_folder, df, mean, std),

    batch_size=batch_size,

    shuffle=False,

    num_workers=0,

    pin_memory=True

)
# useful functions for setting up inference



def sharpen(p,t=0.5):

        if t!=0:

            return p**t

        else:

            return p



def get_classification_preds(net,test_loader):

    test_probability_label = []

    test_id   = []

    

    net = net.cuda()

    for t, (fnames, images) in enumerate(tqdm(test_loader)):

        batch_size,C,H,W = images.shape

        images = images.cuda()



        with torch.no_grad():

            net.eval()



            num_augment = 0

            if 1: #  null

                logit =  net(images)

                probability = torch.sigmoid(logit)



                probability_label = sharpen(probability,0)

                num_augment+=1



            if 'flip_lr' in augment:

                logit = net(torch.flip(images,dims=[3]))

                probability  = torch.sigmoid(logit)



                probability_label += sharpen(probability)

                num_augment+=1



            if 'flip_ud' in augment:

                logit = net(torch.flip(images,dims=[2]))

                probability = torch.sigmoid(logit)



                probability_label += sharpen(probability)

                num_augment+=1



            probability_label = probability_label/num_augment



        probability_label = probability_label.data.cpu().numpy()

        

        test_probability_label.append(probability_label)

        test_id.extend([i for i in fnames])



    

    test_probability_label = np.concatenate(test_probability_label)

    return test_probability_label, test_id
# threshold for classification

threshold_label = [0.50,0.50,0.50,0.50,]
augment = ['null'] #['null', 'flip_lr','flip_ud'] #['null, 'flip_lr','flip_ud','5crop']
# Get prediction for classification model



probability_label, image_id = get_classification_preds(model_classification, testset)

predict_label = probability_label>np.array(threshold_label).reshape(1,4,1,1)



image_id_class_id = []

encoded_pixel = []

for b in range(len(image_id)):

    for c in range(4):

        image_id_class_id.append(image_id[b]+'_%d'%(c+1))

        if predict_label[b,c]==0:

            rle=''

        else:

            rle ='1 1'

        encoded_pixel.append(rle)



df_classification = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
df_classification.head()
# get segmentation model and load weights



# Initialize mode and load trained weights

ckpt_path = "../input/res18fullsize/model_swa.pth"

device = torch.device("cuda")

model_segmentation = Unet("resnet18", encoder_weights=None, classes=4, activation=None)

model_segmentation.to(device)

model_segmentation.eval()

state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

model_segmentation.load_state_dict(state["state_dict"])
models = [model_segmentation] # add other models for ensemble
#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
def post_process(probability, threshold, min_size):

    '''Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored'''

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((256, 1600), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num
# thresholds and min_size for segmentation predictions

# play with them and see how LB changes

threshold_pixel = [0.5,0.5,0.5,0.5,] 

min_size = [200,1500,1500,2000]   
#test time augmentation  -----------------------

def null_augment   (input): return input

def flip_lr_augment(input): return torch.flip(input, dims=[2])

def flip_ud_augment(input): return torch.flip(input, dims=[3])



def null_inverse_augment   (logit): return logit

def flip_lr_inverse_augment(logit): return torch.flip(logit, dims=[2])

def flip_ud_inverse_augment(logit): return torch.flip(logit, dims=[3])



augment = (

        (null_augment,   null_inverse_augment   ),

        (flip_lr_augment,flip_lr_inverse_augment),

        (flip_ud_augment,flip_ud_inverse_augment),

    )
# make preidctions for segmentation model



predictions = []

for i, batch in enumerate(tqdm(testset)):

    fnames, images = batch

    #print('images', images.shape)

    images = images.cuda()

    batch_preds = 0

    probabilities = []

    for model in models:

        model = model.cuda()

        for k, (a, inv_a) in enumerate(augment):

                logit = model(a(images))

                p = inv_a(torch.sigmoid(logit))



                if k ==0:

                    probability  = p**0.5

                else:

                    probability += p**0.5

        probability = probability/len(augment)

        probabilities.append(probability)

        

        batch_preds+=probability

    

    batch_preds = batch_preds.data.cpu().numpy()

    #print(batch_preds.shape)

    for fname, preds in zip(fnames, batch_preds):

        for cls, pred in enumerate(preds):

            #print(cls)

            pred, num = post_process(pred, threshold_pixel[cls], min_size[cls])

            rle = mask2rle(pred)

            name = fname + f"_{cls+1}"

            predictions.append([name, rle])





df_segmentation = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df_segmentation.head()
df= df_segmentation.copy()
# stats for predictions from segmentation model

if 1:

        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)

        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)

        pos1 = ((df['Class']==1) & (df['Label']==1)).sum()

        pos2 = ((df['Class']==2) & (df['Label']==1)).sum()

        pos3 = ((df['Class']==3) & (df['Label']==1)).sum()

        pos4 = ((df['Class']==4) & (df['Label']==1)).sum()



        num_image = len(df)//4

        num = len(df)

        pos = (df['Label']==1).sum()

        neg = num-pos



        print('')

        print('\t\tnum_image = %5d(1801)'%num_image)

        print('\t\tnum  = %5d(7204)'%num)

        print('\t\tneg  = %5d(6172)  %0.3f'%(neg,neg/num))

        print('\t\tpos  = %5d(1032)  %0.3f'%(pos,pos/num))

        print('\t\tpos1 = %5d( 128)  %0.3f  %0.3f'%(pos1,pos1/num_image,pos1/pos))

        print('\t\tpos2 = %5d(  43)  %0.3f  %0.3f'%(pos2,pos2/num_image,pos2/pos))

        print('\t\tpos3 = %5d( 741)  %0.3f  %0.3f'%(pos3,pos3/num_image,pos3/pos))

        print('\t\tpos4 = %5d( 120)  %0.3f  %0.3f'%(pos4,pos4/num_image,pos4/pos))
df_mask = df_segmentation.copy()

df_label = df_classification.copy()
# do filtering using predictions from classification and segmentation models

assert(np.all(df_mask['ImageId_ClassId'].values == df_label['ImageId_ClassId'].values))

print((df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels'] != '').sum() ) #202

df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels']=''
df_mask.to_csv("submission.csv", index=False)
# stats for final submission

if 1:

        df_mask['Class'] = df_mask['ImageId_ClassId'].str[-1].astype(np.int32)

        df_mask['Label'] = (df_mask['EncodedPixels']!='').astype(np.int32)

        pos1 = ((df_mask['Class']==1) & (df_mask['Label']==1)).sum()

        pos2 = ((df_mask['Class']==2) & (df_mask['Label']==1)).sum()

        pos3 = ((df_mask['Class']==3) & (df_mask['Label']==1)).sum()

        pos4 = ((df_mask['Class']==4) & (df_mask['Label']==1)).sum()



        num_image = len(df_mask)//4

        num = len(df_mask)

        pos = (df_mask['Label']==1).sum()

        neg = num-pos



        print('')

        print('\t\tnum_image = %5d(1801)'%num_image)

        print('\t\tnum  = %5d(7204)'%num)

        print('\t\tneg  = %5d(6172)  %0.3f'%(neg,neg/num))

        print('\t\tpos  = %5d(1032)  %0.3f'%(pos,pos/num))

        print('\t\tpos1 = %5d( 128)  %0.3f  %0.3f'%(pos1,pos1/num_image,pos1/pos))

        print('\t\tpos2 = %5d(  43)  %0.3f  %0.3f'%(pos2,pos2/num_image,pos2/pos))

        print('\t\tpos3 = %5d( 741)  %0.3f  %0.3f'%(pos3,pos3/num_image,pos3/pos))

        print('\t\tpos4 = %5d( 120)  %0.3f  %0.3f'%(pos4,pos4/num_image,pos4/pos))