%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai import metrics
path = untar_data(URLs.CIFAR_100);path
path.ls()
data = ImageDataBunch.from_folder(path,valid='test',ds_tfms=get_transforms(do_flip=True),size=224).normalize(imagenet_stats)
data.show_batch(3,figsize=(12,6))
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.load('stage-1');
learn.fit_one_cycle(2,max_lr=slice(0.0001,1e-3))
learn.unfreeze()
learn.fit_one_cycle(3,max_lr=slice(1e-5,1e-4))
learn.unfreeze()
learn.fit_one_cycle(18,max_lr=slice(1e-6,1e-4))
learn.show_results(rows=3,figsize=(12,10))
intrep = ClassificationInterpretation.from_learner(learn)
intrep.most_confused(min_val=3)
intrep.plot_confusion_matrix(dpi=60)