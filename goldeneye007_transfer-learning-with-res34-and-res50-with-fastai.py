# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai import *

from fastai.vision import *
path=Path("../input/intel-image-classification")

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train='seg_train', valid='seg_test',ds_tfms=tfms, size=224).normalize(imagenet_stats)
data.show_batch(rows = 3,figsize=(5,5))
print(data.classes)

print(len(data.train_ds))

print(len(data.valid_ds))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.model_dir ="/tmp/model/"

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
data50 = ImageDataBunch.from_folder(path, train='seg_train', valid='seg_test',ds_tfms=tfms, size=300, bs=32).normalize(imagenet_stats)
learn50 = cnn_learner(data50, models.resnet50, metrics=error_rate)
learn50.model_dir ="/tmp/model/"

learn50.lr_find()

learn50.recorder.plot()
learn50.fit_one_cycle(7)
interp50 = ClassificationInterpretation.from_learner(learn50)



losses50,idxs50 = interp50.top_losses()



len(data50.valid_ds)==len(losses50)==len(idxs50)
interp50.plot_top_losses(9, figsize=(15,11))
interp50.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp50.most_confused(min_val=2)