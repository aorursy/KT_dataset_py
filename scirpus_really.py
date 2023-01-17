# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

%matplotlib inline

# import cv2
from tqdm import tqdm_notebook, tnrange
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/test"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
img_size_ori = 101
img_size_target = 101
im_width = 101
im_height = 101
x_test = np.array([((np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255) for idx in (test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
badimgs = 0
for i,f in enumerate(test_df.index.values):
    if(x_test[i].std()==0):
        print(f)
        badimgs+=1
print(badimgs)
print('Percentage Bad',badimgs/18000)