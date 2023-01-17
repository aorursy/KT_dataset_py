from fastai import *

from fastai.vision import *



import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import auc,roc_curve



from math import floor
path = Path('../input/wangdataset/Images/')
fnames=get_files(path, recurse=True)
fnames[:10]
tfms=get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)
def get_labels(file_path): 

    filename = os.path.basename(file_path)

    label = floor( int(filename.split('.')[0]) / 100 )

    return(label)
data = ImageDataBunch.from_name_func(path, fnames, label_func=get_labels,  size=96, 

                                     bs=64,num_workers=2

                                  ).normalize()
data.show_batch(rows=3, figsize=(8,8))
learner= cnn_learner(data, models.densenet121, metrics=[accuracy], model_dir='/tmp/models/')
learner.lr_find()

learner.recorder.plot()
lr=1e-1

learner.fit_one_cycle(1, lr)
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.freeze_to(-2)

learner.fit_one_cycle(50,slice(1e-3,1e-2))