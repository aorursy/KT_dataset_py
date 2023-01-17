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
import torch

import torch.nn as nn

import numpy as np
data = np.array([[1, 2],

                 [1, 3],

                 [1, 4]]).astype(np.float32)
bn_torch = nn.BatchNorm1d(num_features=2)

data_torch = torch.from_numpy(data)

bn_output_torch = bn_torch(data_torch)

print(bn_output_torch)
help(nnn.B)
import torch

from torch import nn

bn = nn.BatchNorm2d(num_features=2,eps=1e-5, affine=False, track_running_stats=False)#这里的B可以指代三维卷积中的特征位

x = torch.rand(2,3,4,4) #B,C,H,W

official_bn = bn(x)   # 官方代码

print(official_bn)

x1 = x.permute(1, 2, 3, 0).reshape(3, -1) # 对(N, H, W)计算均值方差

mean = x1.mean(dim=1).reshape(1, 3, 1, 1)

# x1.mean(dim=1)后维度为(3,)

std = x1.std(dim=1, unbiased=False).reshape(1, 3, 1, 1)

my_bn = (x - mean)/std

print(my_bn)

print((official_bn-my_bn).sum())  # 输出误差
import torch

from torch import nn

ln = nn.LayerNorm(normalized_shape=[3, 5, 5], eps=1e-8, elementwise_affine=False)

x = torch.rand(2, 3, 5, 5)

official_ln = ln(x)   # 官方代码

print(official_ln)

x1 = x.reshape(2, -1)  # 对（C,H,W）计算均值方差

mean = x1.mean(dim=1).reshape(2, 1, 1, 1)

std = x1.std(dim=1, unbiased=False).reshape(2, 1, 1, 1)

my_ln = (x - mean)/std

print(my_ln)

print((official_ln-my_ln).sum())
import torch

from torch import nn

In = nn.InstanceNorm2d(num_features=5, eps=1e-5, affine=False, track_running_stats=False)

x = torch.rand(2, 3, 5, 5)

official_In = In(x)   # 官方代码

print(official_In)

x1 = x.reshape(2,3, -1)  # 对（H,W）计算均值方差

mean = x1.mean(dim=-1).reshape(2, 3, 1, 1)

std = x1.std(dim=-1, unbiased=False).reshape(2, 3, 1, 1)

my_In = (x - mean)/std

print(my_In)

print((official_In-my_In).sum())
import torch

from torch import nn

gn = nn.GroupNorm(num_groups=2, num_channels=4, eps=1e-5, affine=False)

# 分成了4组，也就是说蓝色区域为（5，5, 5）

x = torch.rand(2, 4, 5, 5)

official_gn = gn(x)   # 官方代码

print(official_gn.shape)

x1 = x.reshape(2,2,4//2,5,5)  # 对（H,W）计算均值方差

x1 = x.reshape(2,2,-1)

mean = x1.mean(dim=-1).reshape(2,2,-1)

std = x1.std(dim=-1, unbiased=False).reshape(2, 2, -1)

my_gn = ((x1 - mean)/std).reshape(2, 2, 2, 5,5)

my_gn = my_gn.reshape(2,4,5,5)

print((official_gn-my_gn).sum())