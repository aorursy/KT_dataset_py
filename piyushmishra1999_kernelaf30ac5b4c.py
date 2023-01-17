from fastai import *

from fastai.vision import *

import os

from os import listdir

%reload_ext autoreload

%autoreload 2

%matplotlib inline

path = "../input/grape/Grape/"

os.listdir(path)
path = Path(path); path
data = ImageDataBunch.from_folder(path, valid_pct = 0.2, size = 224)

data.show_batch(rows = 4)
data = data.normalize()
data.show_batch(rows=3, figsize=(15,11))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(5)
interpretation = ClassificationInterpretation.from_learner(learn)

losses, indices = interpretation.top_losses()

interpretation.plot_top_losses(4, figsize=(15,11))
interpretation.plot_confusion_matrix(figsize=(12,12), dpi=60)
interpretation.most_confused(min_val=2)
learn.model_dir = '/kaggle/working'

learn.save('classification-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion = True)
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=1e-4)
learn.save('classifier-2')
learn.export('/kaggle/working/resnet34-grape.pkl')