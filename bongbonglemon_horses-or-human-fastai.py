%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.metrics import error_rate

from fastai.vision import *

from fastai.basics import *
bs = 64
path = "../input/horses-or-humans-dataset/horse-or-human"

data = ImageDataBunch.from_folder(path, valid="validation")
data.show_batch(rows=3, figsize=(7,6))
data.classes
len(data.classes)
data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4) # number of epochs
learn.model_dir='/kaggle/working/'
learn.save('stage_1')
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses() # in the valid dataset



len(data.valid_ds) == len(losses) == len(idxs)



losses[0:9]



interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
doc(learn.fit_one_cycle)
learn.fit_one_cycle(2, max_lr=slice(1e-7))
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses() # in the valid dataset



len(data.valid_ds) == len(losses) == len(idxs)



losses[0:9]



interp.plot_top_losses(9, figsize=(15,11))