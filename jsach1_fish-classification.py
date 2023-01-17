# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
classes = ["guppy","oscar","swordfish"]
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
bs = 2
fish_images_path = '../input/'

tfms = get_transforms()

data = ImageDataBunch.from_folder(fish_images_path, train='training', valid='validation',ds_tfms=tfms, bs=2, size=128,classes=["guppy","oscar","swordfish"])
data.valid_dl.x[2]

data.show_batch(rows=2, figsize=(15,6))
path_model='/kaggle/working/'

path_input='/kaggle/input/'

learn_resnet34 = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=f'{path_model}')
learn_resnet34.fit_one_cycle(4)