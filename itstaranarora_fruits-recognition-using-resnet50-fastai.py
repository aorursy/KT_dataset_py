import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

from fastai.vision import *
%reload_ext autoreload

%autoreload 2

%matplotlib inline

bs = 64
path = URLs.LOCAL_PATH/'../input/fruits-360_dataset/fruits-360'

tfms = get_transforms(do_flip=False)

np.random.seed(2)

data = ImageDataBunch.from_folder((path), train="Training", valid="Test", ds_tfms=tfms, size=52)

data.show_batch(rows=3, figsize=(5,5))

learn = create_cnn(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(4, max_lr=slice(3e-5,3e-4))
learn.save("4-epoch")
learn.unfreeze()
learn.fit_one_cycle(6, max_lr=slice(3e-5,3e-4))
learn.save("stage-1")