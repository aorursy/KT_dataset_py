import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from fastai import *

from fastai.vision import *
img = open_image(Path('/kaggle/input/br-coins/classification_dataset/all/25_1477287102.jpg'))

print(img.size)

img
train_path = Path('/kaggle/input/br-coins/classification_dataset/all/')

fnames = get_image_files(train_path)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(train_path,

                                   fnames,

                                   pat,

                                   ds_tfms=get_transforms(flip_vert=True, max_zoom=1.0,max_rotate=25, max_lighting=0.1, max_warp=0.1, p_affine=0.75, p_lighting=0.75),# A small portion of the images have the coin near the corner of the image so zooming in can remove alot of signal

                                   size=64,#224,#480

                                   bs=64

                                  ).normalize(imagenet_stats)
data.show_batch(row=3, figsize=(12,12))
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=error_rate)
Model_Path = Path('/kaggle/input/brazillian-coin-fastai-classifier/')

learn.model_dir = Model_Path

learn.load('stage-1');
learn.load('stage-1')

learn.unfreeze()
learn.fit_one_cycle(2)
Model_Path = Path('/kaggle/working')

learn.model_dir = Model_Path

learn.save('stage-2')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()
#looking at the curve, we don't have much we can use for a learning rate

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save('stage-3')
learn.destroy()
data = ImageDataBunch.from_name_re(train_path,

                                   fnames,

                                   pat,

                                   ds_tfms=get_transforms(flip_vert=True, max_zoom=1.0,max_rotate=25, max_lighting=0.1, max_warp=0.1, p_affine=0.75, p_lighting=0.75),

                                   size=224,

                                   bs=64,

                                  ).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=error_rate)
Model_Path = Path('/kaggle/working')

learn.model_dir = Model_Path

learn.load('stage-3');
learn.fit_one_cycle(2)
learn.unfreeze()

lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-6,1e-4))
learn.save('stage-4')
learn.destroy()
data = ImageDataBunch.from_name_re(train_path,

                                   fnames,

                                   pat,

                                   ds_tfms=get_transforms(flip_vert=True, max_zoom=1.0,max_rotate=25, max_lighting=0.1, max_warp=0.1, p_affine=0.75, p_lighting=0.75),

                                   size=480,

                                   bs=64

                                  ).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, pretrained=False, metrics=error_rate)

Model_Path = Path('/kaggle/working')

learn.model_dir = Model_Path

learn.load('stage-4');
learn.fit_one_cycle(6, 1e-4)#
lr_find(learn)

learn.recorder.plot()
learn.save('stage-5')
learn.unfreeze()

learn.fit_one_cycle(1, slice(1e-6,1e-5))
learn.save('final')