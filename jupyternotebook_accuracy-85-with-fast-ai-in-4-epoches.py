#Inspired by fast.ai Practical Deep Learning for Coders(v3) lession 1
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir('../input/car_data/car_data'))
data_dir='../input/car_data/car_data'



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
data.show_batch(rows=3, figsize=(40,40))
print(data.classes)

len(data.classes)
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.model
learn.fit_one_cycle(10)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(30,25))
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(10)
learn.lr_find()


learn.recorder.plot()
learn.unfreeze() 

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(30,25))