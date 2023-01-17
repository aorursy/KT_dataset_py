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
from fastai import *

from fastai.vision import *
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
path = Path("../input/chest_xray/chest_xray/")

path
path.ls()
fnames = get_image_files(path/'train'/'PNEUMONIA')

fnames[:5]
# Training data has 3875 images of Pneumonia Cases

fnames_train_pneumonia = np.array(fnames)

fnames_train_pneumonia.shape
# Training data has 1341 images of Normal Cases

fnames = get_image_files(path/'train'/'NORMAL')

np.array(fnames).shape
# Lets see how many validation folder has

fnames = get_image_files(path/'val'/'NORMAL')

print(np.array(fnames).shape)



fnames = get_image_files(path/'val'/'PNEUMONIA')

print(np.array(fnames).shape)
# Lets see how many test folder has

fnames = get_image_files(path/'test'/'NORMAL')

print(np.array(fnames).shape)



fnames = get_image_files(path/'test'/'PNEUMONIA')

print(np.array(fnames).shape)
np.random.seed(42)

tfms = get_transforms(do_flip=False)



data = ImageDataBunch.from_folder(path, train="train", valid_pct=0.20,

                                  ds_tfms = tfms, classes = ['PNEUMONIA', 'NORMAL'], bs=64, size=224).normalize(imagenet_stats)
print(data.classes)

print(len(data.classes))

print(data.c)
data.show_batch(3, figsize=(12,12))
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir = "/temp/model/")

learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)
interp.plot_top_losses(9, figsize=(12,12))
interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
interp.most_confused(min_val=2)
learn.save('stage-1')
learn.unfreeze()

learn.fit_one_cycle(2)
learn.load('stage-1')

learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(4, figsize=(10,8), heatmap=False)

plt.show()
interp.plot_confusion_matrix(figsize=(10, 8), dpi=60)
learn.show_results()