# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_csv= pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
import seaborn as sns
# Create Class count dataframe
class_count = pd.DataFrame(df.sum(axis=0)).reset_index()
class_count=class_count.rename(columns={"index": "class", 0: "count"})
class_count.drop(class_count.index[0], inplace=True)
# Visualize class distribution as Barchartfig, ax= plt.subplots(figsize=(12,10))
sns.barplot(y="class", x="count",  data=class_count);
train_csv
import matplotlib.pyplot as plt
img_array = plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_0.jpg')
fig = img_array/255
plt.figure(figsize=(10,10))
plt.imshow(fig)
plt.show()
%reload_ext autoreload
%autoreload 2
%matplotlib inline
import pandas as pd
import numpy as np

from fastai.vision import *
from fastai import *
from fastai.callbacks import CSVLogger, SaveModelCallback
import matplotlib.pyplot as plt
import seaborn as sns
import torch
path = Path('/kaggle/input/plant-pathology-2020-fgvc7/')
img_path = Path('/kaggle/input/plant-pathology-2020-fgvc7/images')
data_src = (ImageList.from_df(df=df, path=path, folder='images',suffix=".jpg").
            split_by_rand_pct(0.2)
            .label_from_df(cols=list(class_count['class']),  label_cls=MultiCategoryList, one_hot=True))
data_src
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
data = (data_src.transform(tfms, size=256).databunch().normalize(imagenet_stats))
data.show_batch(4)
f_score = partial(fbeta, thresh=0.45)
learn = cnn_learner(data, models.resnet34, metrics=[accuracy_thresh, f_score],wd=1e-1, callback_fns=[CSVLogger,ShowGraph],path='/kaggle/working')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(20, slice(7e-01))
learn.recorder.plot()
img = open_image("/kaggle/input/plant-pathology-2020-fgvc7/images/Test_487.jpg")
img
pred_class,pred_idx,outputs = learn.predict(img)
pred_class
