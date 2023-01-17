%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision import *

# Function that returns an image from its url



def get_img(img_path): return open_image(img_path)

# Function that displays many transformations of an image

def plots_of_one_image(img_path, tfms, rows=1, cols=3, width=15, height=5, **kwargs):

    img = get_img(img_path)

    [img.apply_tfms(tfms, **kwargs).show(ax=ax) 

     for i,ax in enumerate(plt.subplots(rows,cols,figsize=(width,height))[1].flatten())]           

    

import os

os.listdir() 

os.getcwd()
from glob import glob

list_img_train = sorted(glob('../input/siimdatasetx128/siim-datasetx128/siim-datasetx128/train/128/dicom/*.png'))

list_img_mask_train =  sorted(glob('../input/siimdatasetx128/siim-datasetx128/siim-datasetx128/train/128/mask/*.png'))

list_img_test =  sorted(glob('../input/siimdatasetx128/siim-datasetx128/siim-datasetx128/test/128/dicom/*.png'))



print(len(list_img_train),len(list_img_mask_train),len(list_img_test))
_,axs = plt.subplots(1,4,figsize=(20,10))

for (i,ax),(img_,j) in zip(enumerate(axs),[(get_img(list_img_train[j]),j) 

                                           for j in [2974,9638,9357,8702]]):

    img_.show(ax=ax, title=f'Xray {j}')

    

_,axs = plt.subplots(1,4,figsize=(20,10))

for (i,ax),(img_,j) in zip(enumerate(axs),[(get_img(list_img_mask_train[j]),j) 

                                           for j in [2974,9638,9357,8702]]):

    img_.show(ax=ax, title=f'Mask {j}')
list_img_train[2974]

tfms = [rotate(degrees=(-30,30), p=1.0)]

plots_of_one_image(list_img_train[2974],tfms,padding_mode='border')

plots_of_one_image(list_img_mask_train[2974],tfms,padding_mode='border')
list_img_train[2974]

tfms = [rotate(degrees=(-30,30), p=1.0)]

plots_of_one_image(list_img_train[2974],tfms,padding_mode='reflection')

plots_of_one_image(list_img_mask_train[2974],tfms,padding_mode='reflection')
tfms = [brightness(change=(0.1, 0.9))]

plots_of_one_image(list_img_train[2974],tfms)
tfms = [contrast(scale=(0.5, 2.), p=1.)]

plots_of_one_image(list_img_train[2974],tfms)
fig, axs = plt.subplots(1,3,figsize=(20,5))

for magnitude, ax in zip(np.linspace(-0.05,0.05,5), axs):

    tfms = [jitter(magnitude=magnitude, p=1.)]

    get_img(list_img_train[2974]).apply_tfms(tfms).show(ax=ax,title="magnitude={}".format(magnitude))
tfms = [symmetric_warp(magnitude=(-0.2,0.2), p=1.)]

plots_of_one_image(list_img_train[2974],tfms,padding_mode='zeros')

plots_of_one_image(list_img_mask_train[2974],tfms,padding_mode='zeros')
fig, axs = plt.subplots(1,3,figsize=(20,5))

for scale, ax in zip(np.linspace(1.,2.5,3), axs):

    tfms = [zoom(scale=scale, p=1.)]

    get_img(list_img_train[2974]).apply_tfms(tfms).show(ax=ax,title='scale={}'.format(scale))

    

fig, axs = plt.subplots(1,3,figsize=(20,5))

for scale, ax in zip(np.linspace(1.,2.5,3), axs):

    tfms = [zoom(scale=scale, p=1.)]

    get_img(list_img_mask_train[2974]).apply_tfms(tfms).show(ax=ax,title='scale={}'.format(scale))