%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
help(untar_data)
path = untar_data(URLs.PETS)

path
path.ls()
path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),

                                   size=224, bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),

                                   size=299, bs=bs//2).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.model
learn.fit_one_cycle(5)
learn.save('stage50-1')
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
interp.most_confused(min_val=2)
learn.unfreeze()
learn.load('stage50-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))