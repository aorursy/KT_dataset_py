# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

        

%reload_ext autoreload

%autoreload 2

%matplotlib inline



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *
bs = 64
path = Path('/kaggle/input/100-bird-species/175')
path.ls()
path_valid = path/'valid'

path_train = path/'train'

path_test = path/'test'
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224, bs=bs

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)



print(len(data.classes), data.c) # .c is number of classes for classification problems
#Training the model
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4) # 4 Epochs
learn.model_dir='/kaggle/working/'
learn.save('birds_stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



#Quick check

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(40,40), dpi=400)
interp.most_confused(min_val=2)
#Tweaking our model to make it better
learn.unfreeze()
learn.fit_one_cycle(2)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.load('birds_stage-1')
#Training: resnet50
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=299, bs=bs//2, num_workers=0

                                  ).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
#learn.model_dir='/kaggle2/workingresnet50/'

learn.model_dir='/kaggle/working'
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4)
learn.save('birds_stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
learn.load('birds_stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)