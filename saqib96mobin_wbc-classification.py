%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

import pandas as pd

import os
image_data = Path('../input/dataset2-master/dataset2-master/images/')
image_data.ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(image_data, train='TRAIN', valid='TEST', size = 128, bs=32, num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(12,8))
model_path=Path('/tmp/models/')

learn = cnn_learner(data, models.resnet34, metrics = error_rate, model_dir=model_path)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=1e-2)
learn.save("stage-1")
learn.load("stage-1")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr = slice((1e-4)/2))
learn.save("stage-2")
learn.load("stage-2")
np.random.seed(42)

data = ImageDataBunch.from_folder(image_data, train = 'TRAIN', valid = 'TEST',size = 240, bs=16, num_workers=0).normalize(imagenet_stats)
learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-3, 1e-2))
learn.save("stage-3")
learn.load("stage-3")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-3,1e-2))
learn.save("stage-4")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize = (8,8))