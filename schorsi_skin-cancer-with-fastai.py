import numpy as np

import pandas as pd

import os

import sys

%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai import *

from fastai.vision import *

PATH = Path('/kaggle/input/skin-cancer-malignant-vs-benign/')
data = ImageDataBunch.from_folder(PATH, train="train/",

#                                  valid="train/",

#                                  test="test/",

                                  valid_pct=.3,

                                  ds_tfms=get_transforms(),

                                  size=224,bs=32, 

                                  ).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
learn = cnn_learner(data, models.resnet50, pretrained=False, metrics=error_rate)

Model_Path = Path('/kaggle/input/skin-cancer-classifier-model/')

learn.model_dir = Model_Path

learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(1)
#learn.load('stage-1')
Model_Path = Path('/kaggle/working/')

learn.model_dir = Model_Path

learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_confusion_matrix(figsize=(4,4))
learn.save('Final')