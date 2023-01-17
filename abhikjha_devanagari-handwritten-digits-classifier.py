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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
import numpy as np

import pandas as pd
path = Path('../input/devanagari')

path.ls()
fnames = get_image_files(path/'test/23')

fnames[:5]
open_image(path/'test/23/31991.png').shape
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, train='train', valid='test', size=32, bs=32).normalize(imagenet_stats)
print(data.classes)

print(len(data.classes))

print(data.c)
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir="/tmp/model/")

learn.fit_one_cycle(4)
data.show_batch(3, figsize=(5,5))
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(10,10))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(3)
learn.load('stage-1');

learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(8, max_lr=slice(1e-6, 1e-4))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(12,12))
interp.most_confused(min_val=2)
preds,y, loss = learn.get_preds(with_loss=True)

# get accuracy

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc*100))