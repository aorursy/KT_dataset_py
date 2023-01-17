%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.vision import *



# BATCH SIZE

bs = 64

bs
# Get data paths for base, imgs, and annos

from shutil import copytree

base = '/tmp/stanford-dogs'

copytree('/kaggle/input', base)

path = datapath4file(base)

imgs_path = path/'images/Images'

annos_path = path/'annotations/Annotation'
# Create ImageDataBunch and save before normalization so we don't re-crop the pics

data = ImageDataBunch.from_folder(imgs_path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=0)

data.save('/tmp/save.pkl')
data.show_batch(rows=3, figsize=(7,8))
data.normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()