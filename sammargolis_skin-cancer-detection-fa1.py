import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input/data/data/"))



from fastai.vision import *

from fastai.metrics import error_rate

# Any results you write to the current directory are saved as output.



#start batch size at 64 can reduce based on GPU

bs = 64
%reload_ext autoreload

%autoreload 2

%matplotlib inline
img_dir='../input/data/data/'

path=Path(img_dir)

path
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6,1e-2)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,bs=32, 

                                  num_workers=0).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=slice(1e-4,1e-2))
interp.plot_top_losses(9, figsize=(15,11))

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)