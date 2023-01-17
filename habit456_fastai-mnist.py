%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
path = untar_data(URLs.MNIST_SAMPLE); path
path.ls()
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26, bs=64)
data.show_batch(rows=3, figsize=(7, 6))
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-04, 1e-02))
learn.save('stage-1')
learn.lr_find()

learn.recorder.plot()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(6,7))
interp.plot_confusion_matrix(figsize=(4, 4), dpi=100, title="Confusion Matrix")