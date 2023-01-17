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

from fastai.metrics import error_rate
size = 224

bs = 64

np.random.seed(2)
data_path = '/kaggle/input/dogs-cats-images/dog vs cat/dataset/'
data = ImageDataBunch.from_folder(Path(data_path), train='training_set', valid='test_set',

                                  ds_tfms=get_transforms(), size=size, bs=bs)

data.classes
print(len(data.classes))

data.show_batch(rows=3, figsize=(5,5))
from os.path import expanduser, join, exists

from os import makedirs

cache_dir = expanduser(join('~', '.cache'))

if not exists(cache_dir):

    makedirs(cache_dir)

torch_dir = join(cache_dir, 'torch')

if not exists(torch_dir):

    makedirs(torch_dir)

checkpoints_dir = join(torch_dir, 'checkpoints')

if not exists(checkpoints_dir):

    makedirs(checkpoints_dir)

print(checkpoints_dir)

# # !cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth

!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth
learn = cnn_learner(data, models.resnet34, metrics=accuracy)

# learn = cnn_learner(data, models.resnet50, metrics=accuracy)

# learn.model
learn.fit_one_cycle(3)
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)
interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(5,5), dpi=125)
interp.most_confused(min_val=2)