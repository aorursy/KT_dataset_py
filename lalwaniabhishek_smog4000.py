#Getting all the necessary imports

import numpy as np

import pandas as pd

from pathlib import Path

from fastai import *

from fastai.vision import *

import torch

import os
data_folder = Path("../input/smog4000/smog4000/Smog4000/")

trfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
data = (ImageList.from_folder(data_folder)

        .split_by_folder()

        .label_from_folder()

        .add_test_folder("../input/smog4000/smog4000/Smog4000/test")

        .transform(trfms, size=128)

        .databunch(bs=64, device= torch.device('cuda:0'))

        .normalize())
#Testing the data loader

data.show_batch(3, figsize=(6,6), hide_axis=False)
learn = cnn_learner(data, models.resnet101, metrics=[error_rate, accuracy], model_dir = os.getcwd())
#learn.lr_find(stop_div=False, num_it=200)
#learn.recorder.plot(suggestion = True)

#min_grad_lr = learn.recorder.min_grad_lr
lr = 1e-03

learn.fit_one_cycle(30, slice(lr))
learn.path = Path()
learn.export()
ls