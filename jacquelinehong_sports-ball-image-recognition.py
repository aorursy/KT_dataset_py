import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from fastai.vision import *



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
proj_path = '/kaggle/input/sports-ball-images/'



folders = ['airhockeypucks', 'baseballs', 'basketballs', 'bowlingballs', 'cricketballs', 'footballs', 'golfballs', 'hockeypucks', 'lacrosseballs', 'poolballs', 'rugbyballs', 'soccerballs', 'softballs', 'tennisballs', 'volleyballs']

for i in folders:

    path = Path(proj_path)

    (path/i).mkdir(parents=True, exist_ok=True)

    

p_path = Path(proj_path)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=.2, ds_tfms=get_transforms(), 

                                  size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
# 8 epochs

learn.fit_one_cycle(8)
learn.model_dir = "/kaggle/working"

learn.save('model1_34', return_path=True)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4), wd=.001)
learn.freeze()

learn.lr_find()

learn.recorder.plot()
learn.save('model2_34')
learn.load('model1_34')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(10,10))
img_baseball = open_image(Path(p_path)/'baseball_validation.jpg')

display(img_baseball)



pred_class, pred_idx, outputs = learn.predict(img_baseball)

pred_class