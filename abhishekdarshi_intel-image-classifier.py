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
print(os.listdir("../input/seg_train/seg_train"))
from fastai.vision import *

from fastai.metrics import error_rate
%reload_ext autoreload

%autoreload 2

%matplotlib inline
img_dir = "../input/"
path = Path(img_dir)

path
tfms = get_transforms(do_flip=False)

data = (ImageList.from_folder(path)

        .split_by_folder(train='seg_train', valid='seg_test')

        .label_from_folder()

        .transform(tfms, size=224)

        .databunch())  
data.classes
len(data.classes)
learn = cnn_learner(data, models.resnet34, model_dir = '/tmp/models', metrics=error_rate)
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 11))
interp.most_confused(min_val=2)
learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-5, 1e-3))