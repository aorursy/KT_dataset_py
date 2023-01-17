# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np 

import tensorflow as tf 

import keras 

import seaborn as sns 

import matplotlib.pyplot as plt 

import pickle 

import cv2

import json 

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



sns.set_style("whitegrid")

my_pal = sns.color_palette(n_colors=10)
!ls -GFlash --color ../input/pku-autonomous-driving/
# let us look inside the train data 

train_csv = pd.read_csv("../input/pku-autonomous-driving/train.csv")

train_csv.head()
plt.rcParams["axes.grid"] = False

fig, ax = plt.subplots(figsize = (10,10))

train_ids = train_csv['ImageId'].values

img_name = train_ids[244]

img = load_img('../input/pku-autonomous-driving/train_images/' + img_name + '.jpg')

plt.imshow(img)

plt.show()


fig, ax = plt.subplots(figsize = (10,10))

train_ids = train_csv['ImageId'].values

img_name = train_ids[244]

mask = load_img('../input/pku-autonomous-driving/train_masks/' + img_name + '.jpg')

plt.imshow(mask)

plt.show()
fig, ax = plt.subplots(figsize = (10,10))

plt.imshow(img)

plt.imshow(mask,cmap=plt.cm.viridis, interpolation='none', alpha = 0.7)



plt.show()