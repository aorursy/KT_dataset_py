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
path_img = '/kaggle/input/the-oxfordiiit-pet-dataset/images/'
fnames = get_image_files(path_img)
fnames[:5]
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats);
data.show_batch(rows=3,figsize=(7,6));
print(data.classes)
print(len(data.classes))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn);
interp.plot_top_losses(9,figsize=(16,16));
interp.plot_confusion_matrix(figsize=(12,12),dpi=60);
interp.most_confused(min_val=2);
learn.unfreeze()
learn.fit_one_cycle(2,max_lr=slice(1e-6,1e-4));