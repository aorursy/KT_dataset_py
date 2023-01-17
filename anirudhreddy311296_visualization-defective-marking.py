# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["font.size"] = 15

import random

import cv2

import json

import seaborn as sns

from collections import Counter

from PIL import Image

import math

from collections import defaultdict

from pathlib import Path



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

train_data.shape
train_data.head(3)
train_data = train_data.dropna()

train_data.shape
def rlemarking(rle, imgshape):

    W = imgshape[0]

    H= imgshape[1]

    

    mark= np.zeros( W*H ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    begain = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, first in enumerate(begain):

        mark[int(first):int(first+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mark.reshape(H,W), k=1 ) )



fig=plt.figure(figsize=(20,100))

columns = 4

rows = 25

for i in range(1, 100+1):

    fig.add_subplot(rows, columns, i)

    

    fn = train_data['ImageId_ClassId'].iloc[i].split('_')[0]

    img = cv2.imread( '../input/train_images/'+fn )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mark = rlemarking( train_data['EncodedPixels'].iloc[i], img.shape  )

    img[mark==1,0] = 255

    

    plt.imshow(img)

plt.show()