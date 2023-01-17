%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
from fastai import *
path = untar_data(URLs.MNIST)
path.ls()
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train='training', valid_pct=0.2, seed = 42, bs=60, ds_tfms = tfms, size=28).normalize(imagenet_stats)
data.show_batch(9,figsize=(12,10))
learn = cnn_learner(data,models.resnet18,metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4)