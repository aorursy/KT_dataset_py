%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *
path = Path('../input/boats')
(path).ls()
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=False),

                                  size=128,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.fit_one_cycle(4)
learn.save('stage-1')
lr = 1e-3
learn.unfreeze()
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-2')
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=False),

                                  size=256,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.fit_one_cycle(5, slice(1e-3))
learn.save('stage-3')