# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/seefood/train"))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from fastai.metrics import error_rate

import numpy as np
np.random.seed(42)

data = ImageDataBunch.from_folder('../input/seefood', size=299, bs=11, valid_pct=0.2, ds_tfms=get_transforms()).normalize(imagenet_stats)
data.show_batch(rows=2)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(15, slice(1e-4,1e-2))