%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.datasets import *
from fastai.metrics import *

import numpy as np
import pandas as pd
import os
import re

import matplotlib.pyplot as plt
PATH = "../input/"
path_img = f'{PATH}train/'
os.listdir(PATH)
# os.listdir(path_img)
fnames = get_image_files(path_img)
fnames[:5]
img = plt.imread(f'{fnames[-1]}')
plt.imshow(img);
img.shape
img[:4,:4]
np.random.seed(33)
pattern = re.compile(r'/([^/]+)\.\d+.jpg$')
data = ImageDataBunch.from_name_re(
    path_img, fnames, pattern, ds_tfms=get_transforms(), size=150, bs=32
                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
len(data.classes),data.c
learn = create_cnn(data, models.resnet34, metrics=accuracy, path='./')
learn.fit_one_cycle(1) # Aqui falta LR
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(6,6), dpi=80)
interp.most_confused(min_val=2)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-3))
data = ImageDataBunch.from_name_re(path_img, fnames, pattern, ds_tfms=get_transforms(),
                                   size=150, bs=16).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=accuracy, path='./')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4)
learn.save('stage-1-50')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-5,1e-2))
