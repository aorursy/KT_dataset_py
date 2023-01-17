# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from fastai.vision import *
path = Path('/kaggle/input/intel-image-classification/seg_train/seg_train/')
path.ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,8))
data.classes,data.c,len(data.train_ds),len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics =accuracy)
from google.cloud import bigquery

client = bigquery.Client()
learn.model_dir='/kaggle/working/'
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5,1e-02)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
lr=1e-02
learn.fit_one_cycle(8,max_lr=slice(1e-04,lr/10))
learn.save('save-2')
learn.load('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,15))
interp.plot_top_losses(6,figsize=(16,12))