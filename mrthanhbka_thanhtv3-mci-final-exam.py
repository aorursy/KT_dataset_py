# settings

%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd
path = Path('../input/data/data')

path.ls()
tfms = get_transforms(True,False)

data = ImageDataBunch.from_folder(path,train='train',test='test', valid_pct=0.2,

                                    ds_tfms=tfms, size=20, bs=20).normalize(imagenet_stats)
data = ImageDataBunch.from_folder(path,train='train',test='test', valid_pct=0.2,

                                    ds_tfms=tfms, size=20, bs=20).normalize(imagenet_stats)
data
data.show_batch(rows=3)
model = models.resnet18
data.path = '/tmp/.torch/model'
learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])
learn.lr_find()
learn.recorder.plot()
lr = 1e-02
learn.save("stage-1")
learn.fit_one_cycle(80,slice(lr))
learn.save("state-2")
learn.unfreeze()
learn.fit_one_cycle(4,slice(lr))
accuracy(*learn.TTA())
learn.save("state-3")
learn.load("state-3")