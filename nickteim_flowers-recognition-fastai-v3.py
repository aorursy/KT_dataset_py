from fastai import *

from fastai.vision import *
import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))

from glob import glob

import random

import cv2

import matplotlib.pylab as plt

import random as rand

import keras

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input

from keras.models import Sequential

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from pathlib import Path

from keras.optimizers import Adam,RMSprop,SGD
path =  Path('../input/flowers-recognition/flowers/flowers')
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

                                  ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(10,10))
learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
learn.lr_find()
learn.lr_find()

learn.recorder.plot()
lr = 0.01

learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))
learn.save('stage-3')
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=256, num_workers=0).normalize(imagenet_stats)



learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr=1e-2/2
learn.fit_one_cycle(5, slice(lr))
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.recorder.plot_losses()
learn.save('stage-5')
learn.load('stage-5');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))