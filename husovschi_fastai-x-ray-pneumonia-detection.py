%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.imports import *



import os
data_path = Path('../input/chest_xray/chest_xray')

data_path
np.random.seed(42)

data = ImageDataBunch.from_folder(path=data_path,

                                  train='train',

                                  valid='val',

                                  test='test',

                                  size=224,

                                  bs=64,

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms())

data.normalize(imagenet_stats)

data.show_batch(rows=5)
learner = cnn_learner(data=data,

                      base_arch=models.resnet50,

                      metrics=error_rate,

                      model_dir="/tmp/model/")
learner.fit_one_cycle(4)
learner.unfreeze()
learner.save('stage-1')
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(10, max_lr=slice(3e-6, 3e-5))

learner.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learner)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))