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
from fastai import *

from fastai.vision import *
inputPath = Path('../input')
folder = 'MaineCoon'

file = 'URLs_MaineCoon.txt'
folder = 'NFC'

file = 'URLs_NFC.txt'
folder = 'Siberian'

file = 'URLs_Siberian.txt'
path = Path('data/cats')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images(inputPath/file, dest, max_pics=200)
classes = ['NFC', 'Siberian', 'MaineCoon']

for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=256, num_workers=0).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(rows=3, figsize=(7,8))
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-6))
learn.save('stage-2')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
from fastai.widgets import *
## Manually remove top-losses

# ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)

# ImageCleaner(ds, idxs, path)
## Manually remove duplicates

# ds, idxs = DatasetFormatter().from_similars(learn)

# ImageCleaner(ds, idxs, path)