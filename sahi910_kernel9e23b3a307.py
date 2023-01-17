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
help(untar_data)
import os

print(os.getcwd())
path = Path('../input/rps-cv-images')

path.ls()


np.random.seed(2)
data = ImageDataBunch.from_folder(

    path=path,

    valid_pct=0.5,

    size=224,

    ds_tfms=get_transforms()

)

data.normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learner = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate], model_dir='/tmp/models/')
learner.lr_find()

learner.recorder.plot()
lr = 1e-03

learner.fit_one_cycle(3, max_lr=slice(lr))
learner.save('stage')
learner.recorder.plot_losses()
learner.unfreeze()
learner.fit_one_cycle(4, max_lr=slice(1e-04))