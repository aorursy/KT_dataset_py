# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from glob import glob
import matplotlib.pyplot as plt
import cv2
import skimage
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data_trian_path  = '../input/train/'
# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train_labels.csv')
data_train.info()
data_train['label'].value_counts()
data_train['label'].hist()
print("cancer")
multipleImages = data_train.loc[data_train['label']==1]['id'].values
# multipleImages = glob(data_trian_path + '**')
i_ = 0
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in multipleImages[1:26]:
    file_path = data_trian_path + l +'.tif'
#     print(file_path)
#     print(i_)
    im = cv2.imread(file_path)
#     im = cv2.resize(im, (96, 96)) 
    cv2.rectangle(im, (32,32), (64,64), (0,255,0), 2)
    plt.subplot(5, 5, i_+1) #.set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1
print("no cancer")
multipleImages = data_train.loc[data_train['label']==0]['id'].values
# multipleImages = glob(data_trian_path + '**')
i_ = 0
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in multipleImages[:25]:
    file_path = data_trian_path + l +'.tif'
    im = cv2.imread(file_path)
    cv2.rectangle(im, (32,32), (64,64), (0,255,0), 2)
#     im = cv2.resize(im, (128, 128)) 
    plt.subplot(5, 5, i_+1) #.set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1