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
path = Path('/kaggle/input/facial-age')
path.ls()
tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                      p_affine=1., p_lighting=1.)
data = (ImageList.from_folder(path)
       .split_by_rand_pct(0.2)
       .label_from_folder()
       .transform(tfms,size=224)
       .databunch(bs=20))

data.show_batch()
learn = cnn_learner(data,models.resnet34,metrics=[accuracy])
learn.model_dir='/kaggle/working'
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,slice(1e-1//2,1e-2))
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.save('stage-1')
learn.fit_one_cycle(10,slice(1e-6,1e-4))
learn.save('stage-2')
learn.lr_find()
learn.recorder.plot()
learn.load('stage-2')
learn.fit_one_cycle(30,1e-4)
learn.save('stage-3')
learn.fit_one_cycle(10,1e-4)
learn.save('stage-4')
learn.freeze()
learn.export('/kaggle/working/export.pkl')