# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import openslide # for image import rasterio

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import rasterio

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



ROOT = "/kaggle/input/prostate-cancer-grade-assessment/"

!ls {ROOT}



train = pd.read_csv(ROOT+"train.csv")

test = pd.read_csv(ROOT+"test.csv")

sub = pd.read_csv(ROOT+"sample_submission.csv")
# Print top 3 results 

train.head(5)
train.describe()
train.info()
import matplotlib.pyplot as plt 

import seaborn as sns 

sns.set()

ax = sns.countplot(y = train['isup_grade'])

plt.title("Image distribution")

plt.xlabel('Severity of the cancer (isup_grade)')

plt.ylabel('Image Count')

plt.show()
sns.countplot(y = train["gleason_score"])

plt.tight_layout()

plt.show()
path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

fg = [0, 6, 46, 15, 2 ,32]

for i in range(len(fg)):

    a = path +train['image_id'][fg[i]]+'.tiff'

    src = rasterio.open(a)

    array = src.read(1)

    from matplotlib import pyplot

    plt.figure(figsize= [10,10])

    #plt.subplot(221)

    plt.title('Severity of the cancer: '+str(train['isup_grade'][fg[i]]) )

    pyplot.imshow(array)

    pyplot.show() 

mask_path = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'

fg = [0, 6, 46, 15, 2 ,32]

i = 0

a = mask_path +train['image_id'][fg[i]]+'_mask.tiff'

src = rasterio.open(a)

array = src.read(1)

from matplotlib import pyplot

plt.figure(figsize= [10,10])

#plt.subplot(221)

plt.title('Mask of Severity level cancer: '+str(train['isup_grade'][fg[i]]) )

pyplot.imshow(array, cmap=plt.cm.viridis, interpolation='none', alpha = 0.7)

pyplot.show() 
from matplotlib import pyplot

mask_path = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'

path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

fg = [0, 6, 46, 15, 2 ,32]

for i in range(len(fg)):

    am = mask_path +train['image_id'][fg[i]]+'_mask.tiff'

    srcm = rasterio.open(am)

    arraym = srcm.read(1)

    plt.figure(figsize= [10,10])

    

    

    

    

    

    a = path +train['image_id'][fg[i]]+'.tiff'

    src = rasterio.open(a)

    array = src.read(1)

    plt.title('Mask of Severity level cancer: '+str(train['isup_grade'][fg[i]]) )

    pyplot.imshow(array)

    pyplot.imshow(arraym, cmap=plt.cm.viridis, interpolation='none', alpha = 0.6)

    

    pyplot.show() 