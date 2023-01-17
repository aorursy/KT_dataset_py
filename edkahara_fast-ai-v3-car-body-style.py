%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
bs = 64
path = Path('../input/car_body_style/images')
path
path.ls()
data = ImageDataBunch.from_folder(path, train='train', test='test', valid_pct=0.3, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=4, figsize=(7,6))
print(data.classes)
len(data.classes), data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.model_dir='/kaggle/working/'
learn.save('resnet34-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-2))
data = ImageDataBunch.from_folder(path, train='train', test='test', valid_pct=0.3, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2, num_workers=0).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.model_dir='/kaggle/working/'
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.save('resnet50-1')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-2))