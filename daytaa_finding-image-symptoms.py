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
import cv2

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

flags
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

t4 = train[train['diagnosis']==4].reset_index()['id_code']

t3 = train[train['diagnosis']==3].reset_index()['id_code']

t2 = train[train['diagnosis']==2].reset_index()['id_code']

t1 = train[train['diagnosis']==1].reset_index()['id_code']

t0 = train[train['diagnosis']==0].reset_index()['id_code']
plt.gray()



for i in range(20):

    path = f"../input/train_images/{t4[i]}.png"

    img = cv2.imread(path)

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20, 10))

    



    ax1.imshow(rgb)

    ax2.imshow(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))

    ax3.imshow(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))

plt.close()
plt.gray()



for i in range(20):

    path = f"../input/train_images/{t0[i]}.png"

    img = cv2.imread(path)

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20, 10))

    



    ax1.imshow(rgb)

    ax2.imshow(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))

    ax3.imshow(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
