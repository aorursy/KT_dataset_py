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
from fastai import *

from fastai.vision import *
path = Path('/kaggle/input/song-30-database')

path.ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(path, size=224, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))

#data.show_batch(rows=3,fig_size(7,8),num_workers=4)
dest = Path('/kaggle/working/learn')

dest
data.classes, data.c, len(data.valid_ds), len(data.train_ds)
learn = cnn_learner(data, models.resnet50, metrics=error_rate,model_dir=dest)
learn.fit_one_cycle(5)
learn.save('stage-1')
learn.lr_find()

learn.recorder.plot()
interp=ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.fit_one_cycle(30,max_lr=slice(1e-3, 3e-2))

interp=ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.fit_one_cycle(30,max_lr=slice(1e-3, 3e-2))