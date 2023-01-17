# reload modules before executing user code

%load_ext autoreload

# reload all modules every time before executing the Python code

%autoreload 2

# view plots in notebook

%matplotlib inline
import os

import shutil

import pandas as pd

from fastai.vision import *

from fastai.widgets import DatasetFormatter, ImageCleaner
path = '/kaggle/input/IDC_regular_ps50_idx5'

path
tfms = get_transforms()
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, valid_pct=0.2, size=224)
data.show_batch(rows=3, figsize=(8, 8))
learner = cnn_learner(data, models.resnet34, metrics=accuracy)
learner.model_dir = '/kaggle/working/models'
learner.lr_find()
learner.recorder.plot()
lr = 1e-03
learner.fit_one_cycle(1, lr)