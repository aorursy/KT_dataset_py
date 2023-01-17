%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import h5py 



from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *



import torch

print('pytorch version: ',torch.__version__)

import torch.utils.data as data

import fastai

print('fastai version: ',fastai.__version__)

import torchvision.models

img_dir = '../input/car_data'

path = Path(img_dir)

path.ls()
data = ImageDataBunch.from_folder(f'{path}',valid_pct = 0.2,size = 224,bs = 64).normalize(imagenet_stats)
for classes, numbers in enumerate(data.classes[:15]):

    print(classes,':',numbers)

len(data.classes),data.c
data.show_batch(rows = 3,figsize = (15,15))
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(6)
learn.save('stage-1')
learn.unfreeze()

learn.fit_one_cycle(6)
learn.save('stage-2',return_path=True)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(4, figsize=(14,14),heatmap=False)
interp.most_confused(min_val=2)
input, target = learn.get_preds()

print (top_k_accuracy(input=input, targs=target,k=1))

print (top_k_accuracy(input=input, targs=target,k=3))
