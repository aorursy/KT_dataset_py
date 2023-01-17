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
%reload_ext autoreload
%autoreload 2
%matplotlib inline
!pip install "torch==1.4" "torchvision==0.5.0"
from fastai.vision import *
from fastai.metrics import error_rate
bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
path = Path("../input/the-simpsons-characters-dataset/simpsons_dataset")
path
path.ls()
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(do_flip=False),size=224,bs=64,train='simpsons_dataset',valid_pct=0.2).normalize(imagenet_stats)
data.show_batch(rows = 3, figsize=(8,8))
print(data.classes)
len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(15,15), dpi=60)
