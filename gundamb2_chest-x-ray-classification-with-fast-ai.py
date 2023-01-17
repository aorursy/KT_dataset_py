import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from pathlib import Path
from fastai import *

from fastai.vision import *



path=Path("../input/chest-xray-pneumonia/chest_xray/chest_xray/")
data = ImageDataBunch.from_folder(path,test='test',train='train',valid='val', size=224)

data.show_batch(rows=3)
print(data.classes)

len(data.classes),data.c
from fastai.metrics import error_rate

learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



print(len(data.valid_ds)==len(losses)==len(idxs))



interp.plot_top_losses(9)
interp.plot_confusion_matrix()
interp.most_confused(min_val=2)
learn.unfreeze()
from fastai.vision import get_transforms



transforms=get_transforms(max_rotate=45, max_zoom=1.5, max_lighting=0.5, max_warp=0.3)

data = ImageDataBunch.from_folder(path,test='test',train='train', ds_tfms=transforms, valid='val', size=224)
learn.lr_find()

learn.recorder.plot()

plt.title("Loss Vs Learning Rate")

plt.show()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-3))