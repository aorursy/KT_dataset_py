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
from fastai import *

from fastai.vision import *
path = untar_data(URLs.PETS); path
URLs
URLs.PETS
help(untar_data)
path.ls()
path_img = path / 'images'

path_anno = path/ 'annoatations'
print(path_img); print(path_anno)
fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=64)

data.normalize(imagenet_stats)
data.show_batch(rows = 4, figsize=(7,6))
print(data.classes); print(len(data.classes)); data.c
learn = create_cnn(data, models.resnet34, metrics = error_rate )
learn.fit_one_cycle(4)
learn.save('stage-1')
interpret = ClassificationInterpretation.from_learner(learn)
interpret.plot_top_losses(9, figsize = (15,11))
doc(interpret.plot_top_losses)
interpret.plot_confusion_matrix(figsize = (12, 12), dpi = 60)
interpret.most_confused(min_val = 2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2, max_lr = slice(1e-6, 1e-4))
bs = 64

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms = get_transforms(), size = 320, bs = bs // 2)
learn = cnn_learner(data, models.resnet50, metrics = error_rate)
learn.fit_one_cycle(8, max_lr = slice(1e-03))