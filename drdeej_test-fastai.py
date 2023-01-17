# !pip install fastai

# print(os.listdir("../input"))

# for (dirpath, dirnames, filenames) in os.walk("../input"):

#     print("Directory path: ", dirpath)

#     print("Folder name: ", dirnames)

#     print("File name: ", filenames)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai import *

from fastai.vision import *



# Any results you write to the current directory are saved as output.



boatspath='../input/boats'

flowerspath='../input/flowers-recognition/flowers/flowers'

chestxraypath='../input/chest-xray-pneumonia/chest_xray/chest_xray'



mushrooms='../input/mushrooms-classification-common-genuss-images/mushrooms/Mushrooms'

indianmovieface='../input/indian-movie-face-database-imfdb/'

malaria='../input/cell-images-for-detecting-malaria/cell_images/cell_images'



ctmedical='../input/siim-medical-images'

catslice='../input/ct-slice-localization'

flatsforrent='../input/flats-to-rent-at-budapest'

beeimages='../input/honey-bee-annotated-images'







# data = ImageDataBunch.from_folder(path=chestxraypath, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

# data = ImageDataBunch.from_folder(path=flowerspath,valid_pct=.2, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

data = ImageDataBunch.from_folder(path=boatspath, valid_pct=.2, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

# data = ImageDataBunch.from_folder(path=mushrooms,valid_pct=.2, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

data.classes

data.show_batch(rows=3, figsize=(7,6))
# from PIL import ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True



learn = cnn_learner(data, models.resnet18, metrics=[accuracy,error_rate], model_dir="/kaggle/working")

learn.fit_one_cycle(4)
learn.save('stage-1')
# learn.load('stage-1-50');