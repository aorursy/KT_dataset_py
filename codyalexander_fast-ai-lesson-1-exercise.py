%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.vision import *



# Batch size

bs = 64
help(URLs)
help(untar_data)
path = untar_data(URLs.CIFAR)
path
path.ls()
path_test = path/'test'

path_train = path/'train'
path_test.ls()
path_train.ls()
doc(ImageDataBunch)
data = ImageDataBunch.from_folder(path=path, train='train', valid='test', size=32, bs=bs, ds_tfms=get_transforms(), num_workers=0)

data.normalize(cifar_stats)
data.show_batch(rows=5, figsize=(6,6))
data.classes
doc(ImageDataBunch)
help(data.dl)
help(DatasetType)
data.dl(DatasetType.Train).dataset
data.dl(DatasetType.Valid).dataset
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.fit_one_cycle(4)
learn.fit_one_cycle(4)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-3,1e-2))
learn.fit_one_cycle(4, max_lr=slice(1e-3,1e-2))
learn.fit_one_cycle(4, max_lr=1e-3)
learn.fit_one_cycle(4, max_lr=1e-3)
learn.save('lesson-1-28-epochs')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5))
learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-4))
learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-4))
learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-4))
learn.save('lesson-1-44-epochs')
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=20)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)