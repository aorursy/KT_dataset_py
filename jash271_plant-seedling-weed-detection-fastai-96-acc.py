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

np.random.seed(42)
src=(ImageList.from_folder('../input/v2-plant-seedlings-dataset')
.split_by_rand_pct(valid_pct=0.3,seed=42)
.label_from_folder())

data=(src.transform(get_transforms(),size=224)
.databunch(bs=128).normalize(imagenet_stats))
learn = cnn_learner(data, models.resnet34, metrics=[accuracy], 
                   model_dir=Path("/kaggle/working/"), 
                   path=Path("."))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=slice(2e-3,1e-1))
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12))
interp.most_confused()
learn.show_results()
learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=slice(1e-6,3e-6))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=slice(1e-5,1e-4/6))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3,max_lr=slice(2e-6,4e-6))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12))
interp.most_confused()
learn.show_results()
img=open_image('../input/v2-plant-seedlings-dataset/Black-grass/111.png')
img
learn.predict(img)
