%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

from fastai.vision import *
import pathlib

path = pathlib.Path('/kaggle/input/intel-image-classification')

path_train = path/'seg_train/seg_train'

path_valid = path/'seg_test/seg_test' # labelled set to be used for validation

path_test = path/'seg_pred/seg_test'  # Unlabelled set

bs = 64

size = 150
path_valid.ls()
# data = ImageDataBunch.from_folder(path,

#                                   train=path_train,

#                                   train=path_train,

#                                   #valid_pct=0.2,

#                                   ds_tfms=get_transforms(),

#                                   size=224,

#                                   bs=bs).normalize(imagenet_stats)
np.random.seed(7)

src = (ImageList.from_folder(path)

        .split_by_folder(train = 'seg_train',valid = 'seg_test')

        .label_from_folder()

        #.add_test_folder(test_folder = path_test)

      )  
data = (src.transform(get_transforms(), size=128)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
data.show_batch()
data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)
arch = models.resnet50

learn = cnn_learner(data, arch, metrics = accuracy).to_fp16()
#Need to overwrite the path where model gets stored

learn.model_dir  ='/kaggle/working'
learn.lr_find()

learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-75-rn50')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4, lr/5)) # 2e-6 Try replacing with 1e-4
learn.save('stage-2-75-rn50')
data = (src.transform(get_transforms(), size=size)

        .databunch().normalize(imagenet_stats))



learn = learn.to_fp32()

learn.data = data

data.train_ds[0][0].shape
learn.freeze()

learn.lr_find()

learn.recorder.plot()
lr=1e-2/2

learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-150-rn50')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.recorder.plot_losses()
learn.save('stage-2-150-rn50')