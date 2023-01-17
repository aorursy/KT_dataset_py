# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

import os
path = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')

path.ls()
img = open_image(path/'val'/'NORMAL'/'NORMAL2-IM-1440-0001.jpeg')

print(img.data.shape)

img.show()
torch.Size([3, 1225, 1632])
tfms = get_transforms()
np.random.seed(7)

data = ImageDataBunch.from_folder(path, 

                                  valid='val',

                                  valid_pct=0.2,

                                  size=256,

                                  ds_tfms=tfms).normalize(imagenet_stats)
np.random.seed(7)

data = ImageDataBunch.from_folder(path, 

                                  valid='val',

                                  valid_pct=0.2,

                                  size=256,

                                  ds_tfms=tfms).normalize(imagenet_stats)
np.random.seed(7)

data = ImageDataBunch.from_folder(path, 

                                  valid='val',

                                  valid_pct=0.2,

                                  size=256, 

                                  ds_tfms=tfms).normalize(imagenet_stats)
data.show_batch(3, figsize=(6,6))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(4)
learn.save('../input/chest-xray-pneumonia/chest_xray/chest_xray')
learn.save('xraymodel')
from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server
learn.save("xray", return_path=True)
pwd

from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server