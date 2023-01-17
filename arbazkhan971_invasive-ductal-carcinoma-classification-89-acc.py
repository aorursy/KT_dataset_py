#importing libraries

from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate

import os

import pandas as pd

import numpy as np
x  = "/kaggle/input/breast-histopathology-images"

path = Path(x)

pattern= r'([^/_]+).png$'

fnames=get_files(path, recurse=True)

tfms=get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)

path.ls()
np.random.seed(40)

data = ImageDataBunch.from_name_re(path, fnames, pattern, ds_tfms=tfms, size=50, bs=64,num_workers=4

                                  ).normalize()
data.show_batch(rows=3, figsize=(7,6),recompute_scale_factor=True)

print(data.classes)

len(data.classes)

data.c
data
learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
learn.lr_find()

learn.recorder.plot(suggestions=True)
lr1 = 1e-3

lr2 = 1e-1

learn.fit_one_cycle(1,slice(lr1,lr2))
# lr1 = 1e-3

lr = 1e-1

learn.fit_one_cycle(1,slice(lr))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(1,slice(1e-4,1e-3))
learn.recorder.plot_losses()

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.export()

learn.model_dir = "/kaggle/working"

learn.save("stage-1",return_path=True)