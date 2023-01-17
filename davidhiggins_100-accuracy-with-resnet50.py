# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
import glob

from pathlib import Path

from skimage import io, transform

from fastai.vision import *



train_images = glob.glob('/kaggle/input/fingers/train/*.png')

test_images = glob.glob('/kaggle/input/fingers/test/*.png')



labels = pd.DataFrame({"paths": [ "../.." + img for img in train_images] , "label": [img[-6:-4] for img in train_images]})

labels.to_csv("labels.csv", index=False)
labels = pd.read_csv("labels.csv")



path = Path('/kaggle/working')

tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=224, num_workers=4, valid_pct=0.2).normalize(imagenet_stats)



data.show_batch(rows=3, figsize=(5, 5))
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('resnet-50')
preds, y, losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)

interp.plot_top_losses(9, figsize=(7,7))