# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision import *
tfms = get_transforms(flip_vert=True,

                      max_rotate=45,

                      max_zoom=1.05,

                      max_lighting=0.1,

                      max_warp=0.0)

tfms
!ls /kaggle/input/fruits/fruits-360_dataset/fruits-360/Test
path = Path('/kaggle/input/fruits/fruits-360_dataset/fruits-360/')
src = (ImageList.from_folder(path)

        .split_by_folder(train='Training', valid='Test')

        .label_from_folder())
data = (src

        .transform(tfms, size=50, padding_mode='border')

        .databunch(bs=512)

        .normalize(imagenet_stats))
data.show_batch()
data.c
learn = cnn_learner(data, models.resnet50, metrics=accuracy, path='/kaggle/working', callback_fns=ShowGraph)
learn.lr_find(start_lr=1e-7, end_lr=10, num_it=118*1)
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, 1e-4)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_top_losses(16, figsize=(15,11), heatmap=False)
interp.plot_confusion_matrix(figsize=(24,24), dpi=60)
interp.most_confused()
learn.save('fruits360-r50-stage-1')
learn.load('fruits360-r50-stage-1');
learn.unfreeze()
learn.lr_find(start_lr=1e-7, end_lr=1e-2, num_it=118 * 1, stop_div=False)
learn.recorder.plot(skip_end=30)
learn.fit_one_cycle(10, slice(1e-6, 1e-5))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_top_losses(16, figsize=(15,11), heatmap=False)
interp.plot_confusion_matrix(figsize=(24,24), dpi=60)
interp.most_confused()
learn.save('fruits360-r50-stage-2')
learn.load('fruits360-r50-stage-2');
data = (src

        .transform(tfms, size=100, padding_mode='border')

        .databunch(bs=512)

        .normalize(imagenet_stats))
learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find(start_lr=1e-7, end_lr=1e0, num_it=118 * 1, stop_div=False)
learn.recorder.plot(skip_end=20, suggestion=True)
learn.fit_one_cycle(10, 3e-3)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_top_losses(16, figsize=(15,11), heatmap=False)
interp.plot_confusion_matrix(figsize=(24,24), dpi=60)
interp.most_confused()
learn.save('fruits360-r50-stage-3')
learn.load('fruits360-r50-stage-3');
learn.unfreeze()
learn.lr_find(start_lr=1e-7, end_lr=1e-2, num_it=118 * 1, stop_div=False)
learn.recorder.plot(skip_end=45, suggestion=True)
learn.fit_one_cycle(10, slice(3e-7, 3e-4))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_top_losses(16, figsize=(15,11), heatmap=False)
interp.plot_confusion_matrix(figsize=(24,24), dpi=60)
interp.most_confused()
learn.save('fruits360-r50-stage-4')
learn.export('fruits360-r50.pkl')