!curl -s https://course.fast.ai/setup/colab | bash
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from fastai.vision import *

from fastai.metrics import error_rate



print(os.listdir("../input/cell_images/cell_images"))

path = "../input/cell_images/cell_images"

bs = 64
data = ImageDataBunch.from_folder(path, train = ".",

                                  valid_pct = 0.2,

                                  ds_tfms = get_transforms(flip_vert=True, max_warp=0),

                                  size = 224,

                                  bs = 64,

                                  num_workers = 0

                                ).normalize(imagenet_stats)
data.show_batch(rows = 3, figsize=(7,6))
learn = create_cnn(data, models.resnet50, metrics = error_rate, model_dir = '/tmp/model/')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(7, 1e-02)
learn.save('model-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.load('model-1')
learn.fit_one_cycle(5, max_lr=slice(5e-6,5e-5))
learn.save('model-2')
learn.load('model-2')
intrep = ClassificationInterpretation.from_learner(learn)
intrep.plot_top_losses(9, figsize = (15,11))
intrep.plot_confusion_matrix(figsize =(8,8), dpi = 60)
intrep.most_confused(min_val = 2)