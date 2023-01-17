%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai import metrics
path = untar_data(URLs.PETS);path
path.ls()
path_imgs = path/'images'

path_annotations = path/'annotations'
fnames = get_image_files(path_imgs)
fnames[:5]
random.seed(22)

pat = r'/([^/]+)_\d+.jpg$'
bs=64
data = ImageDataBunch.from_name_re(path_imgs, fnames, pat, ds_tfms = get_transforms(),size=224, bs=bs//2).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))
intrep = ClassificationInterpretation.from_learner(learn)
intrep.most_confused(min_val=2)
learn.show_results(rows=3,figsize=(12,6))