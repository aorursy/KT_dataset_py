# settings

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# load libraries

from fastai import *

from fastai.vision import *



import pandas as pd
size = 16 # ssize of input images

bs = 64 # batch size

tfms = get_transforms()
# Note : Try download data from internet
# Download data

path = untar_data(URLs.CIFAR); path.ls()
# Load data to DataBunch

data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,

                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)

data
data.show_batch(rows=3)
model = models.densenet121
learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])
learn.summary()
learn.lr_find()

learn.recorder.plot()
lr = 1e-2
learn.fit_one_cycle(9,slice(lr))
learn.save("stage-1")
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
lr = 1e-4
learn.fit_one_cycle(9,slice(lr/1e2,lr))
learn.fit(10)
accuracy(*learn.TTA())
learn.save('stage-2')
learn.load('stage-2')
size = 24
# train with the change in images size

data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,

                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)

data
learn.data = data
learn.freeze()
lr = 1e-3
learn.fit_one_cycle(12,slice(lr))
learn.fit(5)
learn.unfreeze()
learn.fit(9)
accuracy(*learn.TTA())
learn.save('stage-3')
learn.load('stage-3')
size = 32
# train with the change in images size

data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,

                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)

data
learn.data = data
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr = 1e-3
learn.fit_one_cycle(12,slice(1))
learn.fit(5)
learn.unfreeze()
learn.fit(9)
accuracy(*learn.TTA())
# Interpretation