# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir('../input/fruits-360_dataset/fruits-360'))
data_dir='../input/fruits-360_dataset/fruits-360'



list = os.listdir(data_dir) 

number_files = len(list)

print(number_files)
path=Path(data_dir)

path
data = ImageDataBunch.from_folder(path,  

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=True,flip_vert=False, max_rotate=90),

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(10,10))
print(data.classes)

len(data.classes)
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.model
learn.fit_one_cycle(3)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(30,25))
interp.most_confused(min_val=1)
learn.unfreeze()
learn.fit_one_cycle(3)
learn.lr_find()
learn.recorder.plot()