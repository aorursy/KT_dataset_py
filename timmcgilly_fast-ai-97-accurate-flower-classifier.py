%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
path = Path("/kaggle/input/")
path.ls()
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224, valid_pct=0.2, num_workers=0, bs=64).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
import torch

torch.cuda.is_available()
learn.fit_one_cycle(4)
learn.save("stage-1-flower_recog")
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
learn.model_dir = "/kaggle/working"

learn.save("stage-2-flower_recog")
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)