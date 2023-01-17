# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai.vision.all import *

from fastai.imports import *

from fastai.vision.data import *

from fastai import *



import fastai

import matplotlib.pyplot as plt
path = Path("/kaggle/input/mechanical-tools-dataset/Mechanical Tools Image dataset-20201009T110652Z-001/Mechanical Tools Image dataset/")

path.ls()
np.random.seed(42)

data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2, item_tfms=RandomResizedCrop(512, min_scale=0.75),

                                    bs=32,batch_tfms=[*aug_transforms(size=256, max_warp=0), Normalize.from_stats(*imagenet_stats)],num_workers=0)
data.show_batch(nrows=3, figsize=(7,8))
%matplotlib inline
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
img = mpimg.imread('../input/mechanical-tools-dataset/Mechanical Tools Image dataset-20201009T110652Z-001/Mechanical Tools Image dataset/Wrench/000197.jpg')

print(img)
implot = plt.imshow(img)
lum_img = img[:,:,0]

plt.imshow(lum_img)
plt.imshow(lum_img, cmap="hot")
imgplot = plt.imshow(lum_img)

imgplot.set_cmap('nipy_spectral')
imgplot = plt.imshow(lum_img)

imgplot.set_cmap('summer')
imgplot = plt.imshow(lum_img)

plt.colorbar()

imgplot.set_cmap('winter')

plt.title("Color Bar")
plt.hist(lum_img.ravel(), bins=100, range=(0.0, 1.0), fc='k', ec='k')
img2 = mpimg.imread('../input/mechanical-tools-dataset/Mechanical Tools Image dataset-20201009T110652Z-001/Mechanical Tools Image dataset/Wrench/000084.jpg')

print(img)
img2plot = plt.imshow(img2)
lum_img2 = img2[1:,0:,0]

plt.imshow(lum_img2)
plt.imshow(lum_img2,cmap="rainbow")
img2plot = plt.imshow(lum_img2)

img2plot.set_cmap('Set3')
img2plot = plt.imshow(lum_img2)

img2plot.set_cmap('bone')
img2plot = plt.imshow(lum_img2)

plt.colorbar()

plt.title("Color Bar")