# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/where_am_i/data/examples/may_the_4_be_with_u/where_am_i/train"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import os

import glob

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt
labelmap=pd.read_csv('../input/mid_term_mapping.txt',names=['place','index'])
labelmap
# 以 "index" 為基準，重新排列

sortlabel=labelmap.sort_values('index')

sortlabel
targetlist=sortlabel.place.values.tolist()

targetlist
sz=10

x=[]

y=[]

for i,c in enumerate(targetlist):

    ff=glob.glob(('../input/where_am_i/data/examples/may_the_4_be_with_u/where_am_i/train/'+c+'/*jpg'))

    for f in ff:

        img=cv2.imread(f,0)

        img200=cv2.resize(img,(sz,sz))

        iii=(np.reshape(img200,(sz,sz,1))).astype(float)

        x.append(iii)

        nn=np.zeros(15,dtype=float)

        nn[i]=1

        y.append(nn)

train_image=np.array(x)

train_target=np.array(y)
train_image
train_target