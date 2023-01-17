# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/ucmerced-landuse-small/ucmerced_landuse_small/ucmerced_landuse_small"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
np.random.seed(999)

path_to_dataset =  "../input/ucmerced-landuse-small/ucmerced_landuse_small/ucmerced_landuse_small/"
data = ImageDataBunch.from_folder(path = path_to_dataset, size=96,bs = 8, num_workers = 0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(10,10))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet18, metrics=[error_rate], model_dir="/tmp/model/")
learn.fit_one_cycle(15) # The parameter within parentheses is for epochs. 

#One epoch is the number of times the algorithm looks at the dataset.
#learn.export('model_ucmerced_landuse.pkl')

learn.export("/kaggle/working/model_ucmerced_landuse.pkl")

path = learn.path

path
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12))
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val = 1)
np.random.seed(10000)

path_to_actor_dataset =  "../input/amit-dharam/amit_dharam/amit_dharam/"
data = ImageDataBunch.from_folder(path = path_to_actor_dataset, size=96,bs = 8, num_workers = 0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(5,5))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet18, metrics=[error_rate], model_dir="/tmp/model/")

learn.fit_one_cycle(1)

learn.save("initial-resnet18")
learn.fit_one_cycle(10)
learn.load("initial-resnet18")

learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
interp.plot_top_losses(9, figsize=(10,10))
learn.lr_find()

learn.recorder.plot()