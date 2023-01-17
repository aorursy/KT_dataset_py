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
# -*- coding:utf-8 -*-

__author__ = 'Leo.Z'



import torch

import time

# 查看torch版本

print(torch.__version__)

# 定义矩阵a和b，随机值填充

a = torch.randn(10000, 1000)

b = torch.randn(1000, 2000)

# 记录开始时间

t0 = time.time()

# 计算矩阵乘法

c = torch.matmul(a, b)

# 记录结束时间

t1 = time.time()

# 打印结果和运行时间

print(a.device, t1 - t0, c.norm(2))   # 这里的c.norm(2)是计算c的L2范数



# 使用GPU设备

device = torch.device('cuda')

# 将ab搬到GPU

a = a.to(device)

b = b.to(device)

# 运行，并记录运行时间

t0 = time.time()

c = torch.matmul(a, b)

t1 = time.time()

# 打印在GPU上运行所需时间

print(a.device, t1 - t0, c.norm(2))



# 再次运行，确认运行时间

t0 = time.time()

c = torch.matmul(a, b)

t1 = time.time()

print(a.device, t1 - t0, c.norm(2))