%cd /
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("kaggle/input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 24
path = Path("/kaggle/input"); path.ls()
np.random.seed(301289)
data = ImageDataBunch.from_folder(path, train='training', valid='validation', size=224, bs=bs, ds_tfms=get_transforms()).normalize(imagenet_stats)
print(data.classes)

len(data.classes),data.c
learn = create_cnn(data, models.resnet34, metrics=[error_rate, accuracy], path='/kaggle')
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused()
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(1) 
learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-3))