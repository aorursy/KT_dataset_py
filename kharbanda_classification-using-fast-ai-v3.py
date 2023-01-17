# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/nonsegmentedv2"))

# Any results you write to the current directory are saved as output.
!cp -r ../input/nonsegmentedv2 .
from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
path = Path('nonsegmentedv2')
fnames = get_image_files(path, recurse=True)
fnames[:5]
np.random.seed(2)
pat = r'\/(\D+)\/\d+.*?.png'
np.random.seed(42)
data = ImageDataBunch.from_name_re(path,fnames,pat, ds_tfms=get_transforms(),valid_pct=0.25,size=224,num_workers=0)
data.normalize(imagenet_stats)
print(data.classes)
len(data.classes),data.c
data.show_batch(rows=3,figsize=(7,6))
learn = create_cnn(data,models.resnet34,metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stg-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12),dpi=60)
interp.most_confused(min_val=2)
learn.lr_find()
learn.recorder.plot()
#learn.load('stg-1')
learn.unfreeze()
learn.fit_one_cycle(4,max_lr=slice(1e-5,1e-3))
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
interp.plot_confusion_matrix(figsize=(12,12),dpi=60)
interp.plot_top_losses(9,figsize=(15,11))
from sklearn import metrics
print(metrics.classification_report(interp.y_true.numpy(), interp.pred_class.numpy(),target_names =data.classes))
!rm -rf nonsegmentedv2
