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
import warnings
warnings.filterwarnings("ignore")
path = Path('/kaggle/input/african-wildlife')
path
data_64 = (ImageList.from_folder(path) 
                .split_by_rand_pct(0.1, seed=33) 
                .label_from_folder()
                .transform(get_transforms(), size=64)
                .databunch(bs=256)
                .normalize(imagenet_stats))
data_64
learn = cnn_learner(data_64, 
                    models.resnet50, 
                    metrics=accuracy, 
                    model_dir='/tmp/model/')
imageData = ImageDataBunch.from_folder(path, valid_pct=0.2, ds_tfms=get_transforms(flip_vert=True, max_rotate=45), size=250).normalize(imagenet_stats)
imageData.show_batch(5, figsize=(15, 11))
learn = cnn_learner(data_64, models.resnet50, pretrained=True, metrics=[accuracy, error_rate])
learn.lr_find() 
learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2)
data_128 = (ImageList.from_folder(path) 
                .split_by_rand_pct(0.1, seed=33) 
                .label_from_folder()
                .transform(get_transforms(), size=128)
                .databunch(bs=128)
                .normalize(imagenet_stats))
learn.data = data_128 
learn.unfreeze() 
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, slice(1e-4,1e-3))
data_256 = (ImageList.from_folder(path) 
                .split_by_rand_pct(0.1, seed=33) 
                .label_from_folder()
                .transform(get_transforms(), size=256)
                .databunch(bs=64)
                .normalize(imagenet_stats))
learn.data = data_256 
learn.unfreeze() 
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 1e-04)