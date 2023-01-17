# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import torch

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

from fastai.vision import *
path = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')
path.ls()
get_image_files(path/'train/NORMAL')[:5]
bs = 64

size = 224

num_workers = 0

tfms = get_transforms()                               

data = (ImageList.from_folder(path)

        .split_by_folder(train='train', valid='val')

        .label_from_folder()

        .transform(get_transforms(do_flip=False),size=224)

        .databunch()

        .normalize() )
print(len(data.train_ds))

print(len(data.valid_ds))
data.classes
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/tmp/models/')
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-6,9e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
interp.most_confused(min_val=2)
learn.export('/export.pkl')
path2 = Path('/')

path2.ls()