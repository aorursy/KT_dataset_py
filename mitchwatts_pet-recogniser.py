import pandas as pd
pd.read_csv('../working/')
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
from fastai import*

from fastai.vision import *
%matplotlib inline

%reload_ext autoreload

%autoreload 2
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
path = untar_data(URLs.PETS); path

path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs

                                  ).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)


learn.save('stage-1')
learn.load('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)

learn.unfreeze()

learn.fit_one_cycle(1)
learn.load('stage-1');

learn.lr_find()

learn.recorder.plot()


learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
# training resnet50