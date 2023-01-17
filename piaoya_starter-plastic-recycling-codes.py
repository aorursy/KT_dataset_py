%reload_ext autoreload

%autoreload 2

%matplotlib inline





from fastai.vision import *

from fastai.metrics import error_rate, fbeta

import pandas as pd

import numpy as np
folder= 'working_dir'
path = Path('../input')

origin= Path('..')

dest = origin/folder



dest.mkdir(parents=True, exist_ok=True)

!cp -r ../input/* {dest}/

path.ls()
dest.ls()
path = Path('../input')
bs=32

tfms=ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0)
np.random.seed(42)

data = (ImageList.from_folder(path)

        .split_by_rand_pct(0.2)

        .label_from_folder()

        .transform(tfms, size=128)

        .databunch())
data.show_batch(rows=3, figsize=(9,7))
classes = data.classes

print(classes)
from fastai.vision.learner import create_cnn,models

from fastai.vision import error_rate
learn = cnn_learner(data, models.resnet50, model_dir = '/tmp/models',  metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, slice(1e-03,4e-3))

learn.save('plastics_save_1', return_path=True)
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6, slice(1e-03,3e-4))

learn.save('plastics_save_2', return_path=True)
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
from sklearn.metrics import classification_report
interp.plot_confusion_matrix()
interp.most_confused()
interp.plot_top_losses(9)
learn.show_results(rows=3, figsize=(10,10))
learn.export('/kaggle/dest')