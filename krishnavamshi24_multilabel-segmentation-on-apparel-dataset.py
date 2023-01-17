

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from fastai import *
from fastai.vision import *

path = Path('/kaggle/input/apparel-dataset/')
path.ls()
img = ImageList.from_folder(path, recurse = True)
img.items.shape
img.open(img.items[10])
np.random.seed(33)
src = (img.split_by_rand_pct(0.2).label_from_folder(label_delim = '_'))
tfms = get_transforms()
data = (src.transform(tfms, size =128).databunch().normalize(imagenet_stats))
data.show_batch(rows = 3, figsize = (15,11))
print(f""" list of classes in the dataset: {data.classes}\n
        number of labels in the dataset: {data.c}\n
        length of training data: {len(data.train_ds)}\n
        length of validation data: {len(data.valid_ds)}""")
acc_02 = partial(accuracy_thresh, thresh = 0.2)
learn = cnn_learner(data, models.resnet34, metrics = acc_02)
learn.model_dir = '/kaggle/working/models'
learn.lr_find()
learn.recorder.plot()
lr = 1e-2
learn.fit_one_cycle(5,slice(lr))
learn.save('stage-1-128')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr = slice(3e-6, 3e-4))
learn.recorder.plot_losses()
learn.save('stage-2-128')
