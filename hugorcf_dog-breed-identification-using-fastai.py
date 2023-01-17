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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
# batch size

bs = 64
# data path

PATH = "../input/"



# path to save model, as input path is read-only

MODEL_PATH = "/tmp/model/"
# GPU required

torch.cuda.is_available()
torch.backends.cudnn.enabled
!head ../input/labels.csv
# transforms applied to the images

tfms = get_transforms(do_flip=True)



# create ImageDataBunch object (images are resized to 'size')

data = ImageDataBunch.from_csv(PATH, folder='train', test='test', suffix='.jpg', ds_tfms=tfms,

                               csv_labels='labels.csv', fn_col=0, label_col=1, size=128, bs=bs)



# normalize

data.normalize(imagenet_stats)
data.show_batch(rows=4, figsize=(12,12))
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir=MODEL_PATH)
learn.model
learn.fit_one_cycle(4)
# save the weights

learn.save('/tmp/model/stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses, idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(24,24), dpi=60)
# combinations of predicted and actual that got wrong the most often

interp.most_confused(min_val=3)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('/tmp/model/stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.recorder.plot_lr(show_moms=True)
learn.recorder.plot_losses()
learn.recorder.plot_losses()
learn.load('/tmp/model/stage-1')
learn.unfreeze()

learn.fit(4, lr=slice(1e-6))
learn.fit(4, lr=slice(1e-6))
# transforms applied to the images

tfms = get_transforms(do_flip=True)



# create ImageDataBunch object (images are resized to 'size')

data = ImageDataBunch.from_csv(PATH, folder='train', test='test', suffix='.jpg', ds_tfms=tfms,

                               csv_labels='labels.csv', fn_col=0, label_col=1, size=128, bs=bs//2)



# normalize

data.normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=accuracy, model_dir=MODEL_PATH)
learn.model
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-3,1e-2))
learn.recorder.plot_losses()
learn = create_cnn(data, models.resnet101, metrics=accuracy, model_dir=MODEL_PATH)
learn.model
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, max_lr=slice(1e-3,1e-2))
learn.recorder.plot_losses()
predictions = learn.get_preds(ds_type=DatasetType.Test)
sample_submission_df = pd.read_csv('../input/sample_submission.csv')

sample_submission_df.head()
submission_df = sample_submission_df.copy()

for i in range(len(submission_df)):

    submission_df.iloc[i, 1:] = predictions[0][i].tolist()

submission_df.head()
# Submission

submission_df.to_csv("submission.csv", index=False)