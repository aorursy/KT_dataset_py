%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

import numpy as np
image_data = Path("../input/cell_images/cell_images")
image_data.ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(image_data, train='.', valid_pct=0.2, 

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),size=128, bs=64,

                                  num_workers=0).normalize(imagenet_stats)
data.classes, data.c
data.train_ds[0][0].shape
data.show_batch(rows=3)
learn = cnn_learner(data, models.densenet161, metrics=accuracy, path='./')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6, max_lr=slice(1e-04,1e-3))
learn.save("stage-1")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6, max_lr=slice(1e-06,1e-05))
learn.save("stage-2")
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(12,12))
interp.plot_confusion_matrix(figsize=(6,6))