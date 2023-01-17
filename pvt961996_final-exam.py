# settings

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# load libraries

from fastai import *

from fastai.vision import *

import pandas as pd
size = 16 # ssize of input images

bs = 32 # batch size

tfms = get_transforms(do_flip=False)
path = Path('../input/data/data')
# Load data to DataBunch

data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,

                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)

data
data.show_batch(rows=3)
model = models.resnet18
data.path = '/tmp/.torch/models'

learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])
learn.summary()
learn.lr_find()

learn.recorder.plot()
learn.save("stage-1")
lr = 2e-2
learn.fit_one_cycle(4,slice(lr))
learn.unfreeze()
lr = lr /100

learn.fit_one_cycle(4,slice(lr))
accuracy(*learn.TTA())
learn.save("stage-2")
size = 28
# Load data to DataBunch

data = ImageDataBunch.from_folder(path,train='train',test='test', valid_pct=.2,

                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)

data
learn.data = data
learn.freeze()

learn.lr_find()

learn.recorder.plot()
lr = 1e-2
learn.fit_one_cycle(5,slice(lr))
learn.unfreeze()
lr = lr /100

learn.fit_one_cycle(5,slice(lr))
accuracy(*learn.TTA())
learn.save('stage-3')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)