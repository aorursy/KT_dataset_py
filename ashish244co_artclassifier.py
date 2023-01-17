# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai import *
from fastai.vision import *
from fastai.metrics import *
!pip install fastai
path = Path('/kaggle/input/best-artworks-of-all-time/images/images/')
src = ImageList.from_folder(path).split_by_rand_pct(0.2).label_from_folder()
src
tfms = get_transforms()
data = (src.transform(tfms, size=112).databunch(bs=32).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(5,5))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics=error_rate,model_dir="/tmp/model/")
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-3,1e-2))
learn.save('stage1')
learn.load('stage1')
learn.recorder.plot_losses()
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-5))
learn.save('stage-2')
learn.freeze()
data2 = (src.transform(tfms, size=224).databunch(bs=32).normalize(imagenet_stats))
learn.data=data2
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-4,1e-3))
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-4,1e-3))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-5))
learn.show_results()
learn.save('stage3')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(20,20))
interp.plot_top_losses(9,heatmap=True)