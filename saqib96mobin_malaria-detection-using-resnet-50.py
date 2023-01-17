%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

import pandas as ps

import numpy as np
image_data = Path("../input/cell_images/cell_images")
image_data.ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(image_data, train='.', valid_pct=0.2,ds_tfms=get_transforms(flip_vert=True, max_warp=0), size=128, bs=64, num_workers=0).normalize(imagenet_stats)
data.classes, data.c
data.train_ds[0][0].shape
data.show_batch(rows=3)
model_path = Path('/tmp/models/')

learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir=model_path)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-02,1e-01))
learn.save("stage-1")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-05,1e-04))
learn.save("stage-2")
learn.load("stage-2")
np.random.seed(42)

data = ImageDataBunch.from_folder(image_data, train='.', valid_pct=0.2,ds_tfms=get_transforms(flip_vert=True, max_warp=0), size=224, bs=64, num_workers=0).normalize(imagenet_stats)
learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-03,1e-02))
learn.save("stage-3")
learn.load("stage-3")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-5, 1e-4))
learn.save("stage-4")
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix()