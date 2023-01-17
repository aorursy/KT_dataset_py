# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')



train.head()
train_images_dcm_dir = '../input/siim-isic-melanoma-classification/train/'

train_images_jpeg_dir = '../input/siim-isic-melanoma-classification/jpeg/train/'
dcm = pydicom.dcmread(train_images_dcm_dir + list(train['image_name'])[10] + '.dcm').pixel_array



jpeg = plt.imread(train_images_jpeg_dir + list(train['image_name'])[10] + '.jpg')

f,ax = plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(dcm)

ax[1].imshow(jpeg)



ax[0].set_title('Image from DCM')

ax[1].set_title('JPEG Image')
dcm2 = cv2.cvtColor(dcm, cv2.COLOR_YCrCb2BGR)
f,ax = plt.subplots(1,3,figsize=(15,10))

ax[0].imshow(dcm)

ax[1].imshow(dcm2)

ax[2].imshow(jpeg)



ax[0].set_title('Image from DCM')

ax[1].set_title('DCM image after color conversion')

ax[2].set_title('Corresponding JPEG image')