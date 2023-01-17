# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from fastai.metrics import error_rate
from fastai.vision import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
path = '/kaggle/input/chest-xray-pneumonia/chest_xray/'
os.listdir(path)
tfms = get_transforms(max_rotate=1, max_zoom=.1)
data = ImageDataBunch.from_folder(path, train='train', valid='test', ds_tfms=tfms, size=128, bs=32).normalize(imagenet_stats)
data.batch_stats
data.show_batch(figsize=(10,10), rows=3)
data.classes
learner = cnn_learner(data, models.resnet101, metrics=[error_rate, Recall(), Precision()], callback_fns=ShowGraph, model_dir='/tmp/model/')
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(5)
