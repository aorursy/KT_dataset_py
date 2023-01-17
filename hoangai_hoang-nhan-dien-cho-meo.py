# settings

%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd

tfms = get_transforms(True,False)
path = Path('../input/catsdogs/')

path.ls()
data =  ImageDataBunch.from_folder(path,train='training_set',test='test_set.zip', valid_pct=0.2,

                                    ds_tfms=tfms, size=200, bs=20).normalize(imagenet_stats)
data
data.show_batch(rows=3)
model = models.resnet18
data.path = '/tmp/.torch/dogsandcats'
learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])
learn.summary()
learn.lr_find()
learn.recorder.plot()
lr = 1e-02
learn.fit_one_cycle(4,slice(lr))
learn.save("stage-1")

learn.fit_one_cycle(1,slice(lr))
learn.load("stage-1")

lr = 1e-03
learn.fit_one_cycle(2,slice(lr))
learn.save("stage-2")
learn.unfreeze()
lr = lr /100

learn.fit_one_cycle(4,slice(lr))
learn.save("stage-2")
learn.unfreeze()
lr = 1e-03

learn.fit_one_cycle(4,slice(lr))
accuracy(*learn.TTA())
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_top_losses(9, figsize=(15,11))