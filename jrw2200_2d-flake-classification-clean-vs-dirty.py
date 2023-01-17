import numpy as np

import os

import pandas as pd

import matplotlib.pyplot as plt

import shutil



from shutil import unpack_archive

from subprocess import check_output



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from fastai.vision import *
path = '/kaggle/images/'

shutil.os.mkdir(path)

shutil.os.mkdir(path + 'clean')

shutil.os.mkdir(path + 'dirty')
path = '/kaggle/input/exfoliated-flakes-of-twodimensional-crystals'

fileList = os.listdir(path)

for f in fileList:

    if set(f[:-4].split('_')).intersection({'used', 'clean'}):

        shutil.copy(path + '/' + f, '/kaggle/images/clean')

    else:

        shutil.copy(path + '/' + f, '/kaggle/images/dirty')
data = ImageDataBunch.from_folder('/kaggle/images', train = '.', valid_pct=0.1, ds_tfms=get_transforms(), size=256, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=2, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.load('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()