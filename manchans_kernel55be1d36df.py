# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

!ls ../input
!head ../input/train_ship_segmentations_v2.csv
!pwd
from fastai import *

from fastai.vision import *

# from fastai.torch_imports import *

# from fastai.transforms import *

# from fastai.conv_learner import *

# from fastai.model import *

# from fastai.dataset import *

# from fastai.sgdr import *

# from fastai.plots import *
df_labels = pd.read_csv('../input/train_ship_segmentations_v2.csv')
df_labels.head()
idx = df_labels.ImageId

idx.value_counts()[:5]
df_labels['hasShip'] = df_labels['EncodedPixels'].apply(lambda x: 0 if pd.isnull(x) else 1)
df_labels.head()
#Add a column with counts of no of times ImageId is repeating

df_labels['count'] = df_labels.groupby('ImageId')['ImageId'].transform('count')
df_labels.columns
df_labels.head()
# Make count 0 if no ship present. For count more than 1 multiple ships would be present

df_labels.loc[df_labels['hasShip'] == 0, 'count'] = 0 

# df1.loc[df1['stream'] == 2, 'feat'] = 10

df_labels.shape
df_labels['count'][:15]
df_c = df_labels[['ImageId','count']]

df_c.shape
df_c = df_c.drop_duplicates().reset_index()

df_c.shape
df_labels.loc[:5, 'EncodedPixels']
df_ls = df_c.sample(n=10000, replace = False)
df_ls.shape
test_imgs = os.listdir("../input/test_v2")

train_imgs = os.listdir('../input/train_v2')
test_imgs[:5], train_imgs[-5:]
train_path = '../input/train_v2/'

test_path = '../input/test_v2/'
imgs = []

for i in range(9):

    image = mpimg.imread(train_path+train_imgs[i])

    imgs.append(image)
len(train_imgs), len(test_imgs)
train_imgs_s = [train_imgs[i] for i in df_ls.index]
_, axs = plt.subplots(3, 3, figsize=(18, 18))

axs = axs.flatten()

for img, ax in zip(imgs, axs):

    ax.imshow(img)

plt.show()
bs = 64
path = Path('../input')

path
path.ls()
path_img = path/'train_v2'
# fnames = get_image_files(path_img)

# fnames[:5]
df_ls.to_csv('labels.csv', columns = ['ImageId', 'count'], index=False)
df_ls.shape
!head labels.csv
data = ImageDataBunch.from_csv("", '../input/train_v2', valid_pct = 0.2, size=128, delimiter=',',

    ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_warp=0.)).normalize(imagenet_stats)
# data = ImageDataBunch.from_df(path, df_ls, 'train_v2', valid_pct = 0.2, label_col='hasShip', size=128,

#     ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(10,8))
data.train_ds, data.valid_ds
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-3))
learn.save('stage-10K')
!ls models/
learn.load('stage-10K')
img = learn.data.valid_ds[2][0]
learn.data.valid_ds.items[0]
img = learn.data.valid_ds[0]

img
imgs = {}

for i in range(10):

    img = learn.data.valid_ds[i][0]

    fname =learn.data.valid_ds.items[i]

    imgs[fname] = img

imgs
result = {}

for name,img in imgs.items():

    r = learn.predict(img)

    result[name] = r
for name,res in result.items():

    print(name, ":", res)

im = open_image(test_path+test_imgs[3])

test_imgs[3] 
im
learn.predict(im)
test_r = {}

for i in range(9):

    image = open_image(test_path+test_imgs[i+10])

    r = learn.predict(image)

    test_r[test_imgs[i+10]]=r
for name,r in test_r.items():

    print(name, ":", r)
im=open_image('../input/test_v2/d7ad50e7b.jpg')

im
imgs = []

for i in range(9):

    image = mpimg.imread(test_path+test_imgs[i+10])

    imgs.append(image)
_, axs = plt.subplots(3, 3, figsize=(18, 18))

axs = axs.flatten()

for img, ax in zip(imgs, axs):

    ax.imshow(img)

plt.show()