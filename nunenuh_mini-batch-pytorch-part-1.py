# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!ls -al /kaggle/input/flower_data/flower_data/train/10
%matplotlib inline
import PIL
import PIL.Image
import torch
import numpy as np
import matplotlib.pyplot as plt
path = '/kaggle/input/flower_data/flower_data/train/10/'
filename = 'image_07086.jpg'
pf = path + filename
img = PIL.Image.open(pf)
img_arr = np.array(img)
h,w,c = img_arr.shape
# bikin vector pake torch
vc = torch.rand(10)
print(vc)
# bikin matric pake torch
mx = torch.rand(3,3)
# print(mx)
plt.imshow(mx, cmap='gray')
mx*5
#membuat vector
ar = [1,4,5]
vc = torch.tensor(ar)
print(vc)
#membuat matrix
ar = [
    [11,12,13],
    [21,22,23],
    [31,32,33]
]
mx = torch.tensor(ar)
print(mx)
#akses semua baris pada kolom ke satu
a = mx[:,0]
print(a)

#akses baris ke satu untuk semua kolom
b = mx[0,:]
print(b)

#akses baris pertama pada kolom pertama
c = mx[0,0]
print(c)
#membuat tensor
ar = [
    [
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ],
    [
        [0,255,0],
        [0,0,255],
        [255,0,0]
    ],
    [
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ]
]
tnsr = torch.tensor(ar)
print(tnsr)
plt.imshow(tnsr, cmap='gray')
