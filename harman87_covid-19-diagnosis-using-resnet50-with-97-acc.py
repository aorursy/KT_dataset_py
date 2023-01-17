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
from fastai.vision import *

from fastai.metrics import error_rate

from torchtext.utils import *

import warnings

warnings.filterwarnings("ignore")
# Root directory for dataset

path = Path('/kaggle/input/sarscov2-ctscan-dataset/')  ## setting the path to the train images

path.ls()
#transformations

tfms_tra,tfms_val=get_transforms(flip_vert=True, do_flip=False, max_rotate=15, max_zoom=1.0, max_lighting= 0.5, max_warp=0.4, p_affine=1., p_lighting=1.)

tfms=(tfms_tra, tfms_val)

#batch_size

bs = 16
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.3,

        ds_tfms=tfms, bs=bs, size=224, num_workers=8).normalize(imagenet_stats)
data.train_ds
data.valid_ds
data.show_batch(3, figsize=(6,6), hide_axis=True)
data.classes
learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])
learn.fit_one_cycle(10)
learn.unfreeze()
learn.fit_one_cycle(8, max_lr=1.91E-05)