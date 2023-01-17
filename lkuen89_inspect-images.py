# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

!ls ../input/severstal-steel-defect-detection



BASE_DIR = '../input/severstal-steel-defect-detection'



# Any results you write to the current directory are saved as output.



images_are_grayscale = True
# get mean train image

mean_train_image = np.zeros((256, 1600, 3), np.float32)

imnum = 1

for imname in os.listdir(os.path.join(BASE_DIR, 'train_images')):

    im = np.array(Image.open(os.path.join(BASE_DIR, 'train_images', imname))).astype(np.float32)

    if np.any(im[:,:,0] != im[:,:,1]) or np.any(im[:,:,1] != im[:,:,2]):

        images_are_grayscale = False

    mean_train_image = im + mean_train_image * (imnum-1)/imnum

    imnum += 1

plt.figure()

plt.imshow(mean_train_image/np.max(mean_train_image))
# get mean train image

mean_test_image = np.zeros((256, 1600, 3), np.float32)

imnum = 1

for imname in os.listdir(os.path.join(BASE_DIR, 'test_images')):

    im = np.array(Image.open(os.path.join(BASE_DIR, 'test_images', imname))).astype(np.float32)

    if np.any(im[:,:,0] != im[:,:,1]) or np.any(im[:,:,1] != im[:,:,2]):

        images_are_grayscale = False

    mean_test_image = im + mean_test_image * (imnum-1)/imnum

    imnum += 1



plt.figure()

plt.imshow(mean_test_image/np.max(mean_test_image))
print(images_are_grayscale)
diff_im = mean_train_image/np.max(mean_train_image) - mean_test_image/np.max(mean_test_image)

diff_im += np.min(diff_im)

diff_im = diff_im / np.min(diff_im)

plt.figure()

plt.imshow(diff_im)