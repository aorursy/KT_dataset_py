# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
path = Path('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images')

path.ls()
(path/'TRAIN').ls()
data = ImageDataBunch.from_folder(path, train='TRAIN', test='TEST', valid_pct=0.20,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data
data.show_batch(rows=3, figsize=(10,10))
data.c, len(data.train_ds), len(data.valid_ds), len(data.classes), len(data.test_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.model_dir=Path('/kaggle/working')

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(20,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=100)
interp.most_confused(min_val = 2)
help(interp.confusion_matrix)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save('model_resnet34')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=100)
interp.most_confused()
learn
doc(learn.predict)
doc(ImageDataBunch.from_folder)