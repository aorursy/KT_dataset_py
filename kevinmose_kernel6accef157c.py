# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F
import matplotlib.pyplot as plt

%matplotlib inline



def imshow(img):

  img = img/2 + 0.5

  plt.show(np.transpose(img, (1,2,0)))
""" 

  Double conv 2D in UNET architecture.

  Each Double Conv represents :

  2 x Conv -> Batch_normalization -> ReLU

"""

class DoubleConv2D(nn.Module):

  def __init__(self, in_channels, out_channels):

    super(DoubleConv2D, self).__init__()

    self.conv = nn.Sequential(

        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

        nn.BatchNorm2d(out_channels),

        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),

        nn.BatchNorm2d(out_channels),

        nn.ReLU(inplace=True)

    )

  

  def forward(self,x):

    return self.conv(x)
""""

  Downscaling with maxpool of kernel size (2,2)

  and followed by double conv

"""

class DownScaling(nn.Module):

  def __init__(self, in_channels, out_channels):

    super(DownScaling, self).__init__()

    self.pooling_double_conv = nn.Sequential(

        nn.MaxPool2d(2),

        DoubleConv2D(in_channels, out_channels)

    )



  def forward(self, x):

    return self.pooling_double_conv(x)
"""

  Upscaling + Double conv

"""

class UpScaling(nn.Module):

  def __init__(self, in_channels, out_channels):

    super(UpScaling, self).__init__()

    self.up = nn.ConvTranspoee2d(in_channels//2, in_channels//2, kernel_size=2, strides=2)

    self.dConv = DoubleConv2D(in_channels, out_channels)

  

  def forward(self, x, y):

    #upsampling x

    x = self.up(x)

    #Compute difference vector of W and H for x and y

    #input type : [C, H, W]

    heightDiff = y.size()[2] - x.size()[2]

    widthDiff = y.size()[3] - x.size()[3]

    #convert differrence to tensor

    heightTensor = torch.tensor(heightDiff)

    widthTensor = torch.tensor(widthDiff)

    #Pads the input tensor boundaries with some values

    x = F.pad(x1, [heightTensor//2, heightTensor - heightTensor//2,

                   widthTensor//2, widthTensor - widthTensor//2])

    #Concatenates [y, x] with dimension 1

    output = torch.cat([y, x], dim=1)

    return self.dConv(output)
"""

  Output convolution

"""

class OutputConv2D(nn.Module):

  def __init__(self, in_channels, out_channels):

    super(OutputConv2D, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  

  def forward(self, x):

    return self.conv(x)
class UNet(nn.Module):

  def __init__(self, n_input, n_output):

    super(UNet, self).__init__()

    self.n_channels = n_input

    self.n_classes = n_output



    self.inputConv = DoubleConv2D(n_input,64)

    self.ds1 = DownScaling(64, 128)

    self.ds2 = DownScaling(128, 256)

    self.ds3 = DownScaling(256, 512)

    self.ds4 = DownScaling(512, 512)



    self.ups1 = UpScaling(1024, 256)

    self.ups2 = UpScaling(512, 128)

    self.ups3 = UpScaling(256, 64)

    self.ups4 = UpScaling(128, 64)

    self.outputConv = OutputConv2D(64, n_classes)



  def forward(self, x):

    x1 = self.inputConv(x)        

    x2 = self.ds1(x1)             # 

    x3 = self.ds2(x2)

    x4 = self.ds3(x3)

    x5 = self.ds4(x4)

    x = self.ups1(x5, x4)

    x = self.ups2(x, x3)

    x = self.ups3(x, x2)

    x = self.ups4(x, x1)

    output = self.outputConv(x)

    return output
from torchvision import transforms, datasets, models