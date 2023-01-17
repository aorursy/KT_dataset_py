# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from fastai.vision import *
from pathlib import Path
import os
import gc


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = Path('/kaggle/input/plant-pathology-2020-fgvc7/')
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')
sample_df = pd.read_csv(path/'sample_submission.csv')
train_df.head()
test_df['image_id'] = 'images/' + test_df['image_id'] + '.jpg'
test_data = ImageList.from_df(test_df,path)
tfms = get_transforms(flip_vert=True,max_zoom=1.2,max_lighting=0.1)
src = (ImageImageList.from_csv(path,'train.csv',folder='images',suffix='.jpg').split_by_rand_pct(0.2).label_from_df(cols=[1,2,3,4]).add_test(test_data))
train_data = (src.transform(tfms,size=(64,64)).databunch().normalize(imagenet_stats))

#del train_df
#del test_data
#del tfms
#del test_df
#gc.collect()
learn = cnn_learner(train_data,models.densenet161,metrics=[accuracy],wd=1e-1)
learn.fit_one_cycle(3)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(30,max_lr=slice(1e-3, 1e-3/5))
preds, y = learn.get_preds(DatasetType.Test)
sample_df.iloc[:,1:] = preds.numpy()
sample_df.to_csv('submission.csv', index=False)

