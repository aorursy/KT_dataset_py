from fastai.vision import *

from fastai import metrics
path = untar_data(URLs.CIFAR);path
path.ls()
data = ImageDataBunch.from_folder(path,valid='test',ds_tfms=get_transforms(),size=224)
data.show_batch(3,figsize=(12,6))
learn = cnn_learner(data, models.resnet50,metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.fit_one_cycle(2,max_lr=slice(1e-5,1e-3))
learn.show_results(rows=3,figsize=(14,9))