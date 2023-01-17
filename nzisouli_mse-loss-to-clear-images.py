# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pathlib import Path

import fastai

from fastai.vision import *

from fastai.callbacks import *
path = Path("/kaggle/input/dance-images")

path_cl= path/"clear"

path_bl= path/"blurry"
bs,size=8, 512

arch = models.resnet34

src = ImageImageList.from_folder(path_bl).split_by_rand_pct(0.1, seed=42)
def get_data(bs,size):

    data = (src.label_from_func(lambda x: path_cl/x.name)

           .transform(get_transforms(max_zoom=0), size=size, tfm_y=True)

           .databunch(bs=bs, num_workers=0).normalize(imagenet_stats, do_y=True))



    #data.c = 3

    return data
data = get_data(bs,size)

data.show_batch(4)
y_range = (-3.,3.)

loss = MSELossFlat()
learn = unet_learner(data, arch, blur=True, norm_type=NormType.Weight,

                         self_attention=True, y_range=y_range, loss_func=loss, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
lr = 1e-3

learn.fit_one_cycle(3, lr)
learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-4,lr))

learn.show_results(rows=1, imgsize=5)
bs,size=1, 1024
data = get_data(bs, size)

learn.data = data

gc.collect()
learn.freeze()

learn.fit_one_cycle(3, lr)
gc.collect()

learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-4,lr))

learn.show_results(imgsize=5)
gc.collect()

learn.save('mse-loss')