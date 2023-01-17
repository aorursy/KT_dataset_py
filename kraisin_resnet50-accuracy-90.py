%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
import os
os.listdir('../input/stanford-car-dataset-by-classes-folder/car_data/car_data')
base_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/'
data = ImageDataBunch.from_folder(base_dir, train='train', valid='test', ds_tfms=get_transforms(), size=512, bs=30)
data.normalize(imagenet_stats)
data.show_batch(rows=3, fig_size=(40,40))
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(10, figsize=(30,30))
learn.model_dir='/kaggle/working/'
learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(5e-5, 5e-4))
learn.lr_find()
learn.recorder.plot()
learn.save('stage-2')
