# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
batch_size = 16
path = untar_data(URLs.PETS);path
path.ls
path_img = path/'images'
os.listdir(path_img)[:5]
pat = r'/([^/]+)_\d+.jpg$'
fnames = get_image_files(path_img)
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=batch_size

                                  ).normalize(imagenet_stats)
data
data.show_batch(rows=3,figsize=(7,6))
learn = create_cnn(data,models.resnet34,metrics=error_rate)
learn.save('/kaggle/working/resnet34-10epoch')