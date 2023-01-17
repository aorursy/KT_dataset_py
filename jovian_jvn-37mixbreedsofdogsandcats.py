%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
doc(untar_data)
path = untar_data(url = URLs.PETS) 

path
path.ls()
anno_path = path/'annotations'

img_path = path/'images'
fnames = get_image_files(img_path)

fnames[:5]
pat = re.compile(r'/([^/]+)_\d+.jpg$')

data = ImageDataBunch.from_name_re(img_path, fnames, pat, ds_tfms=get_transforms(), size=224, bs=64).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(4)
learn.save('1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(20,20))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.lr_find()
learn.recorder.plot()
learn.load('1')
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(3e-6, 3e-4))
data = ImageDataBunch.from_name_re(img_path, fnames, pat, ds_tfms=get_transforms(), size=299, bs=32).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8)
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(3e-3,3e-2))