!pip install fastai2 --quiet
#Load the dependancies
from fastai2.vision.all import *

import numpy as np
import pandas as pd
import os
import h5py
import nilearn as nl
from nilearn import image, datasets, plotting
from nilearn.image import get_data
from random import randint

matplotlib.rcParams['image.cmap'] = 'gray'
source = Path("../input/trends-assessment-prediction")
files = os.listdir(source)
print(files)
fnc = pd.read_csv(f'{source}/train_scores.csv')
fnc.head()
train_files = get_files(source/'fMRI_train')
train_files
test_files = get_files(source/'fMRI_test')
test_files
test_img = train_files[0]
test_img = h5py.File(test_img, 'r')
test_img
test_img.keys()
test_img['SM_feature']
test_image = train_files[0]
t = h5py.File(test_image, 'r')['SM_feature'][()]
x_axis = t[:,:,19].transpose(1,2,0)
plt.imshow(x_axis[:, :,28])
@delegates(subplots)
def show_images(ims, nrows=1, ncols=None, titles=None, cmap=None, **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`"
    if ncols is None: ncols = int(math.ceil(len(ims)/nrows))
    if titles is None: titles = [None]*len(ims)
    axs = subplots(nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip(ims, titles, axs): show_image(im, ax=ax, title=t, cmap=cmap)
x_list = []
for i in range(52):
    x_axis = t[:,:,i].transpose(1,2,0)
    x_ = x_axis[:, :,0]
    x_list.append(x_)
show_images(x_list, nrows=4, cmap=plt.cm.Set1)
y_list = []
for i in range(63):
    y_axis = t[:, :, i, :].transpose(1, 2, 0)
    y_ = y_axis[:, :,0]
    y_list.append(y_)
show_images(y_list, nrows=4, cmap=plt.cm.Set1)
z_list = []
for i in range(52):
    z_axis = t[:, i, :, :].transpose(1, 2, 0)
    z_ = z_axis[:, :,0]
    z_list.append(z_)
show_images(z_list, nrows=4, cmap=plt.cm.Set1)
train_path = source/'fMRI_train'
train_path
def mat_x(fn):
    file = int(fn.Id)
    im = f'{train_path}/{file}.mat'
    t = h5py.File(im, 'r')['SM_feature'][()]
    idx_3 = randint(0, 52)
    x_axis = t[:, :, :, idx_3].transpose(1, 2, 0)
    return x_axis[:, :, 0]
def mat_y(fn):
    file = int(fn.Id)
    im = f'{train_path}/{file}.mat'
    t = h5py.File(im, 'r')['SM_feature'][()]
    idx_2 = randint(0, 62)
    y_axis = t[:, :, idx_2, :].transpose(1, 2, 0)
    return y_axis[:, :, 0]
def mat_z(fn):
    file = int(fn.Id)
    im = f'{train_path}/{file}.mat'
    t = h5py.File(im, 'r')['SM_feature'][()]
    idx_1 = randint(0, 51)
    z_axis = t[:, idx_1, :, :].transpose(1, 2, 0)
    return z_axis[:, :, 0]
age_ = fnc['age'].unique()
dom1_ = fnc['domain1_var1'].unique()
dom2_ = fnc['domain1_var2'].unique()
dom3_ = fnc['domain2_var1'].unique()
dom4_ = fnc['domain2_var2'].unique()
blocks = (
          ImageBlock(cls=PILImage),
          ImageBlock(cls=PILImage),
          ImageBlock(cls=PILImage),
          CategoryBlock(vocab=age_),
          CategoryBlock(vocab=dom1_),
          CategoryBlock(vocab=dom2_),
          CategoryBlock(vocab=dom3_),
          CategoryBlock(vocab=dom4_)
          )
         
getters = [
           mat_x,
           mat_y,
           mat_z,
           ColReader('age'),
           ColReader('domain1_var1'),
           ColReader('domain1_var2'),
           ColReader('domain2_var1'),
           ColReader('domain2_var2')
          ]

trends = DataBlock(blocks=blocks,
              getters=getters,
              item_tfms=Resize(128),
              n_inp=3
              )
trends.summary(fnc)
dls = trends.dataloaders(fnc, bs=4)
dls.show_batch(max_n=4)