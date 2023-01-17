!pip install fastai2 -q
#Load the dependancies
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *

import seaborn as sns
import numpy as np
import pandas as pd
import os
import cv2
import openslide

sns.set(style="whitegrid")
sns.set_context("paper")

matplotlib.rcParams['image.cmap'] = 'ocean_r'
source = Path("../input/prostate-cancer-grade-assessment")
files = os.listdir(source)
files
train = source/'train_images'
mask = source/'train_label_masks'
train_labels = pd.read_csv(source/'train.csv')
train_labels.head()
def view_image(folder, fn):
    if folder == train:
        filename = f'{folder}/{fn}.tiff'
    if folder == mask:
        filename = f'{folder}/{fn}_mask.tiff'
    file = openslide.OpenSlide(str(filename))
    t = tensor(file.get_thumbnail(size=(255, 255)))
    if folder == train:
        show_image(t)
    if folder == mask:
        show_image(t[:,:,0])
view_image(train, '0005f7aaab2800f6170c399693a96917')
view_image(mask, '0005f7aaab2800f6170c399693a96917')
def view_images(file, mask, fn):
    ima = f'{file}/{fn}.tiff'
    msk = f'{mask}/{fn}_mask.tiff'
    ima_file = openslide.OpenSlide(str(ima)); ima_t = tensor(ima_file.get_thumbnail(size=(255, 255)))
    ima_msk = openslide.OpenSlide(str(msk)); msk_t = tensor(ima_msk.get_thumbnail(size=(255, 255)))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20, 6))
    s1 = show_image(ima_t, ax=ax1, title='image')
    s2 = show_image(msk_t[:,:,0], ax=ax2, title='mask')
    s3 = plt.hist(msk_t.flatten()); plt.title('mask histogram')
    plt.show()
view_images(train, mask, '06636cdd43041e78141f2f5069fa62d5')
view_images(train, mask, '0d3159cd1b2495cc82637ececf63ed41')
view_images(train, mask, '08134913a9aa1d541f719e9f356f9378')
from fastai2.medical.imaging import *
@patch
@delegates(show_image)
def show(self:PILImage, scale=True, cmap=plt.cm.ocean_r, min_px=None, max_px=None, **kwargs):
    px = tensor(self)
    if min_px is not None: px[px<min_px] = float(min_px)
    if max_px is not None: px[px>max_px] = float(max_px)
    show_image(px, cmap=cmap, **kwargs)
def selective_mask(file, mask, fn, min_px=None, max_px=None):
    ima = f'{file}/{fn}.tiff'
    msk = f'{mask}/{fn}_mask.tiff'
    ima_file = openslide.OpenSlide(str(ima)); ima_t = tensor(ima_file.get_thumbnail(size=(255, 255)))
    ima_msk = openslide.OpenSlide(str(msk)); msk_t = tensor(ima_msk.get_thumbnail(size=(255, 255)))
    msk_pil = PILImage.create(msk_t[:,:,0])
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (20, 6))
    s1 = show_image(ima_t, ax=ax1, title='image')
    s2 = show_image(msk_t[:,:,0], ax=ax2, title='mask')
    s3 = msk_pil.show(min_px=min_px, max_px=max_px, ax=ax3, title=f'selective mask: min_px:{min_px}')
    s4 = plt.hist(msk_t.flatten()); plt.title('mask histogram')
    plt.show()
selective_mask(train, mask, '08134913a9aa1d541f719e9f356f9378', min_px=None, max_px=None)
selective_mask(train, mask, '08134913a9aa1d541f719e9f356f9378', min_px=1, max_px=None)
selective_mask(train, mask, '08134913a9aa1d541f719e9f356f9378', min_px=2, max_px=None)
selective_mask(train, mask, '08134913a9aa1d541f719e9f356f9378', min_px=3, max_px=None)
selective_mask(train, mask, '08134913a9aa1d541f719e9f356f9378', min_px=4, max_px=None)
msk = f'{mask}/08134913a9aa1d541f719e9f356f9378_mask.tiff'
ima_msk = openslide.OpenSlide(str(msk)); msk_t = tensor(ima_msk.get_thumbnail(size=(255, 255)))
msk_pil = PILImage.create(msk_t[:,:,0])
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 6))
s1 = msk_pil.show(min_px=None, max_px=None, ax=ax1, title='original mask')
s2 = msk_pil.show(min_px=1, max_px=2, ax=ax2, title='1 and 2')
s3 = msk_pil.show(min_px=2, max_px=3, ax=ax3, title='2 and 3')
s4 = msk_pil.show(min_px=3, max_px=4, ax=ax4, title='3 and 4')
s4 = msk_pil.show(min_px=4, max_px=5, ax=ax5, title='4 and 5')
plt.show()
train_labels[train_labels.values == '08134913a9aa1d541f719e9f356f9378']
isup_0 = train_labels[train_labels.isup_grade == 0]
isup_0[:1]
selective_mask(train, mask, '0005f7aaab2800f6170c399693a96917', min_px=None, max_px=None)
isup_5 = train_labels[train_labels.isup_grade == 5]
isup_5[:1]
selective_mask(train, mask, '00928370e2dfeb8a507667ef1d4efcbb', min_px=None, max_px=None)
def custom_img(fn):
    fn = f'{train}/{fn.image_id}.tiff'
    file = openslide.OpenSlide(str(fn))
    t = tensor(file.get_thumbnail(size=(255, 255)))
    img_pil = PILImage.create(t)
    return img_pil
def show_selective(p, scale=True, cmap=plt.cm.ocean_r, min_px=None, max_px=None):
    px = tensor(p)
    if min_px is not None: px[px<min_px] = float(min_px)
    if max_px is not None: px[px>max_px] = float(max_px)
    return px
def custom_selective_msk(fn):
    fn = f'{mask}/{fn.image_id}_mask.tiff'
    file = openslide.OpenSlide(str(fn))
    t = tensor(file.get_thumbnail(size=(255, 255)))[:,:,0]
    ts = show_selective(t, min_px=None, max_px=None)
    return ts
blocks = (ImageBlock,
          ImageBlock)

getters = [
           custom_img,
           custom_selective_msk
          ]
prostate = DataBlock(blocks=blocks,
                 getters=getters,
                 item_tfms=Resize(128))

j = prostate.dataloaders(train_labels, bs=16)
j.show_batch(max_n=12, nrows=2, ncols=6)
def custom_selective_msk(fn):
    fn = f'{mask}/{fn.image_id}_mask.tiff'
    file = openslide.OpenSlide(str(fn))
    t = tensor(file.get_thumbnail(size=(255, 255)))[:,:,0]
    ts = show_selective(t, min_px=1, max_px=None)
    return ts
blocks = (ImageBlock,
          ImageBlock)

getters = [
           custom_img,
           custom_selective_msk
          ]
prostate = DataBlock(blocks=blocks,
                 getters=getters,
                 item_tfms=Resize(128))

j = prostate.dataloaders(train_labels, bs=16)
j.show_batch(max_n=12, nrows=2, ncols=6)
def custom_selective_msk(fn):
    fn = f'{mask}/{fn.image_id}_mask.tiff'
    file = openslide.OpenSlide(str(fn))
    t = tensor(file.get_thumbnail(size=(255, 255)))[:,:,0]
    ts = show_selective(t, min_px=None, max_px=None)
    return ts
blocks = (ImageBlock,
          MaskBlock,
          CategoryBlock)

getters = [
           custom_img,
           custom_selective_msk,
           ColReader('isup_grade')
          ]
prostate = DataBlock(blocks=blocks,
                 getters=getters,
                 item_tfms=Resize(224))

j = prostate.dataloaders(train_labels, bs=16)
j.show_batch(max_n=4)