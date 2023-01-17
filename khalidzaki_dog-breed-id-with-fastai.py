import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from fastai.vision import *

from fastai.metrics import error_rate, accuracy



print(os.listdir("../input"))

%reload_ext autoreload

%autoreload 2

%matplotlib inline

PATH = "../input/"

MODEL_PATH = "/tmp/model/"
df = pd.read_csv('../input/labels.csv')

df.head(10)
print(df['breed'].value_counts().sort_values(ascending=False))

plt.figure(figsize=(12,8))

plt.hist(df['breed'].value_counts().sort_values(ascending=False))

plt.show()
tfms = get_transforms(max_rotate=25); len(tfms)

data = ImageDataBunch.from_csv(PATH, folder='train', test='test', suffix='.jpg', ds_tfms=tfms,

                               csv_labels='labels.csv', fn_col=0, label_col=1, 

                               size=128, bs=64).normalize(imagenet_stats)
data.show_batch(rows=4, figsize=(12,8))

print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=[accuracy], model_dir=MODEL_PATH)

learn.model

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(3e-2,1e-2))

learn.save('/tmp/model/stage-1-34')

interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))

interp.most_confused(min_val=2)

#learn.load('/tmp/model/stage-1')

learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

learn.save('/tmp/model/stage-2-34')

data = ImageDataBunch.from_csv(PATH, folder='train', test='test', suffix='.jpg', ds_tfms=tfms,

                               csv_labels='labels.csv', fn_col=0, label_col=1, 

                               size=224, bs=16).normalize(imagenet_stats)
data.show_batch(rows=4, figsize=(12,8))

learn = cnn_learner(data, models.resnet50, metrics=[accuracy], model_dir=MODEL_PATH)

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-3,1e-2))

learn.save('/tmp/model/stage-1-50')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(3, max_lr=slice(1e-5,1e-4))

learn.save('/tmp/model/stage-2-50')

predictions = learn.get_preds(ds_type=DatasetType.Test)

predictions[0][0]

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.head()
submission = sample_submission.copy()

for i in range(len(submission)):

    submission.iloc[i, 1:] = predictions[0][i].tolist()

submission.head()