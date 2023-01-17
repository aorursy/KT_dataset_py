# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pathlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import gc

print(os.listdir("../input"))



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *



# Any results you write to the current directory are saved as output.
DATA_DIR='../input/data/synthetic_digits'
os.listdir(f'{DATA_DIR}')
data = ImageDataBunch.from_folder(DATA_DIR, 

                                  train="imgs_train",

                                  valid="imgs_valid",

                                  #valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=150,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6,1e-2)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, max_lr=slice(1e-6,1e-4 ))
learn.save('stage-2')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, max_lr=slice(1e-6,1e-4 ))
learn.save('stage-3')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, 1e-6)
learn.save('stage-4')
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)