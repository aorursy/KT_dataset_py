%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import os

import pandas as pd
bs = 32 # batch size

sz=224 # image size
base_image = Path('../input/dataset2-master/dataset2-master/images/')

base_image/'TRAIN'

data = ImageDataBunch.from_folder(path=base_image,train='TRAIN',valid='TEST',size=sz,bs=bs,num_workers=0).normalize(imagenet_stats)
data.show_batch()
print(data.classes)

len(data.classes),data.c
model_path=Path('/tmp/models/')

learn = create_cnn(data, models.resnet34, metrics=error_rate,model_dir=model_path)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=1e-2)
learn.save('stage-1-224')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=slice(1e-5,1e-4))
learn.save('stage-2-224')
learn.load('stage-1-224')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)