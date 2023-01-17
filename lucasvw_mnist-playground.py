%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
path = untar_data(URLs.MNIST_TINY)
path.ls()
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=96, num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(5,5))
print(data.classes)
learn = create_cnn(data, models.resnet34, metrics=[error_rate, accuracy])
learn.lr_find()
learn.recorder.plot()
learn.save('stage-0')
learn.fit_one_cycle(4, max_lr=1e-04)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(5,5))
learn.load('stage-0');
learn.fit_one_cycle(4, max_lr=1e-03)
learn.load('stage-0');
learn.fit_one_cycle(4, max_lr=1e-01)
learn.load('stage-0');

learn.fit_one_cycle(4, max_lr=3e-02)
learn.load('stage-0');

learn.fit_one_cycle(4)
learn.load('stage-0');

learn.fit_one_cycle(10, max_lr=3e-02)
learn.load('stage-0');

learn.fit_one_cycle(20, max_lr=3e-02)