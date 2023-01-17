import numpy as np

import pandas as pd

import os

import zipfile

import matplotlib.pyplot as plt

%matplotlib inline

import cv2

import skimage.io

from PIL import Image
BASE_PATH = '../input/prostate-cancer-grade-assessment'



# cvs files

train = pd.read_csv(f'{BASE_PATH}/train.csv').set_index('image_id')

# test = pd.read_csv(f'{BASE_PATH}/test.csv').set_index('image_id')

# submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv').set_index('image_id')



# image and mask directories

test_dir  = f'{BASE_PATH}/test_images'

data_dir = f'{BASE_PATH}/train_images'

mask_dir  = f'{BASE_PATH}/train_label_masks'
level = 1   # [0,1,2]

bins=256
# select image_id

# image_id = train.index[2]   # train index

image_id = '4517c109e23cf3b572373db82b519303'   # image_id
train.loc[image_id,:]
biopsy = skimage.io.MultiImage(os.path.join(data_dir, f'{image_id}.tiff'))

maskfile = skimage.io.MultiImage(os.path.join(mask_dir, f'{image_id}_mask.tiff'))



plt.figure(figsize=(12,10))

plt.subplot(1,2,1)

plt.title('image_id:{}\nisup_grade:{}'

          .format(image_id, train.loc[image_id, 'isup_grade']))

plt.imshow(biopsy[level])

plt.subplot(1,2,2)

plt.imshow(maskfile[level][:,:,0])

plt.title('data_provider:{}\ngleason_score:{}'

          .format(train.loc[image_id, 'data_provider'], train.loc[image_id, 'gleason_score']))

plt.colorbar()

plt.show()
print(image_id, biopsy[level].shape, train.loc[image_id, 'data_provider'], 

      'isup_grade :', train.loc[image_id, 'isup_grade'], train.loc[image_id, 'gleason_score'])
def color_hist(img, maskno):

    img = img.reshape(-1,3)

    plt.hist(img, color=["red", "green", "blue"], histtype="step", bins=bins)

    plt.title('image_id:{}\ndata_provider:{}\nmask label:{}'

          .format(image_id, train.loc[image_id, 'data_provider'], maskno))

    plt.show()
idx = maskfile[level][:,:,0]>0

color_hist(biopsy[level][idx], 'non-zero')

for i in np.unique(maskfile[level][:,:,0]):

    idx = maskfile[level][:,:,0]==i

    color_hist(biopsy[level][idx], i)