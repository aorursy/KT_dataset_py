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
!ls /kaggle/input/dogs-vs-cats-redux-kernels-edition
!unzip /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip

!unzip /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip
!ls
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate

from pathlib import Path

import pandas as pd

import numpy as np

import os

import warnings

warnings.filterwarnings('ignore')
os.getcwd()
path = Path()

print(path.ls())

train_path = path/'train'

test_path = path/'test'

print(train_path, test_path)
fnames = get_image_files(train_path)

fnames[:5]
np.random.seed(2)

#pat = r'/([^/]+).\d+.jpg$'

pat = r"([a-z]+).\d+.jpg$"
bs = 64 #Batch size

data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=get_transforms(),test='test', size=224, bs=bs

                                  ).normalize(imagenet_stats)

data
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
!find . -name *.pth -print
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.load('stage-1');

learn.lr_find(stop_div=False, num_it=200)
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))
learn.recorder.plot_losses()
learn.recorder.plot_lr()
learn.save('stage-2')
preds, _ = learn.get_preds(DatasetType.Test)
print(preds.size())

print(preds)
labels = np.argmax(preds, axis=1)

labels
input_path = Path('/kaggle/input/dogs-vs-cats-redux-kernels-edition')

test_df = pd.read_csv(input_path/'sample_submission.csv')

test_df.head()
ids = [int(file.stem) for file in data.test_ds.x.items]

ids
submission = pd.DataFrame({'id':ids,'label':labels})

submission.head()
submission = submission.sort_values(by=['id'])
submission.to_csv('submission.csv', index=False)