%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai.metrics import error_rate



import os

print(os.listdir("../input/cell_images/cell_images"))
bs = 64 

img_dir= '../input/cell_images/cell_images/'

path = Path(img_dir)

path.ls()
data = ImageDataBunch.from_folder(path, train = ".", valid_pct=0.2, 

                                  ds_tfms = get_transforms(flip_vert=True), 

                                  size = 224, bs=bs).normalize(imagenet_stats)
print(f"Classes: \n {data.classes}")
data.show_batch(rows =4, figsize=(20,20))
learn = cnn_learner(data, models.resnet18, metrics = error_rate, model_dir ="/tmp/model/")
learn.model
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4)

learn.save('Stage1')
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(1e-5, 1e-3))
learn.save('Stage1')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(12, figsize = (20,20), heatmap=False)
interp.plot_confusion_matrix(figsize = (4,4), dpi=60)