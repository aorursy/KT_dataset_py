# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from IPython.display import FileLink



# Any results you write to the current directory are saved as output.
from pathlib import Path

from fastai import *

from fastai.vision import *
!pip freeze | grep fastai
data_folder = Path("../input")

data_folder.ls()
train_img = (ImageList.from_folder(path=data_folder)

        .split_by_folder('seg_train', 'seg_test')

        .label_from_folder()

        .add_test_folder('seg_pred')

        .transform(get_transforms(), size=150)

        .databunch(path='.', bs=32)

        .normalize(imagenet_stats)

       )
train_img
train_img.show_batch()
learn = cnn_learner(train_img, models.resnet152, metrics=[accuracy])
learn.lr_find()
learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(10, lr)

# Public LB : 0.94
learn.save('RN152-BS32-E10-LR01-stage1')
learn.recorder.plot_losses()
learn.recorder.plot_lr()
learn.recorder.plot_metrics()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9)
interp.plot_confusion_matrix()
preds,_ = learn.get_preds(ds_type=DatasetType.Test)

preds.shape
classes = preds.argmax(1)

classes.shape, classes.min(), classes.max()
classes
test_img_names = learn.data.test_ds.items
test_img_names
test_img_names = [i.name for i in test_img_names]
learn.data.classes
test_df = pd.DataFrame({'image_name': test_img_names, 'label': classes})
test_df.head()
test_df['label'].value_counts()
test_df.to_csv('sub_RN152-BS32-E10-LR01-stage1.csv', index=False)
FileLink('sub_RN152-BS32-E10-LR01-stage1.csv')

# Public LB : 0.944292237442922 | CV : 0.929333 
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-5, 8e-5))
learn.save('RN152-BS32-E10-LR01-stage2')
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
inter = ClassificationInterpretation.from_learner(learn)
inter.plot_top_losses(9)
inter.plot_confusion_matrix()
preds,_ = learn.get_preds(ds_type=DatasetType.Test)

preds.shape

classes = preds.argmax(1)

classes.shape, classes.min(), classes.max()

test_df['label'] = classes
test_df.head()
test_df['label'].value_counts()
test_df.to_csv('sub_RN152-BS32-E10-LR01-stage2.csv', index=False)
FileLink('sub_RN152-BS32-E10-LR01-stage2.csv')

# Public LB : 0.9447 | CV : 0.929000