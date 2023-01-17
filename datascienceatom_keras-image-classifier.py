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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

import os
import random
import gc
train_dir = '../input/train'
test_dir = '../input/test'

train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]
train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]

test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_dogs[:2000] + train_cats[:2000]
random.shuffle (train_imgs)

del train_dogs
del train_cats
gc.collect()

import matplotlib.image as mpimg
for ima in train_imgs[0:3]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.show()
nrows = 150
ncolumns = 150
channels = 3
def read_and_process_image(list_of_imgs):
 
   
    X = [] 
    Y = [] 

    for image in list_of_imgs:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
    
        if 'dog' in image:
            Y.append(1)
        elif 'cat' in image:
            Y.append(0)
        
    return X,Y     




