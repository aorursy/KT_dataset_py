%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai.metrics import error_rate

from fastai.text import *

import zipfile

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = Path('../input/cell_images/cell_images')
np.random.seed(2)

data = ImageDataBunch.from_folder(path, train=path,valid_pct = 0.2, size=224, bs=64,ds_tfms = get_transforms()).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
learn = cnn_learner(data, models.resnet101, metrics=accuracy, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6,1e-3)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,max_lr=1e-6)
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11),heatmap=False)
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)