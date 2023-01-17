# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install "torch==1.4" "torchvision==0.5.0"
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.basics import *
from fastai import *
from fastai.vision import *
path=Path('/kaggle/input/10-monkey-species/training/training/')
training = Path('/kaggle/input/10-monkey-species/training/training/')
validation = Path('/kaggle/input/10-monkey-species/validation/validation/')
path.ls()
np.random.seed(42)
data = ImageDataBunch.from_folder('/kaggle/input/10-monkey-species/', train="/training/training/", valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data,models.resnet50,metrics=error_rate,model_dir = '/kaggle/working/models/')
data
data.classes, data.valid_ds.classes, len(data.train_ds), len(data.valid_ds)
learn.fit_one_cycle(1)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6,max_lr=slice(1e-5,5e-3))
learn.save('stage-2')
learn.load('stage-2')
interp=ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
data.show_batch(rows=3,figsize=(7,8))
interp.plot_top_losses(12,figsize=(20,15))

