#importing libraries
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
x  = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'
path = Path(x)
path.ls()
np.random.seed(40)
data = ImageDataBunch.from_folder(path, train = '.', valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=224,
                                  num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6),recompute_scale_factor=True)

print(data.classes)
len(data.classes)
data.c
data
learn = cnn_learner(data, models.resnet50, metrics=[accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
learn.lr_find()
learn.recorder.plot(suggestions=True)
lr1 = 1e-3
lr2 = 1e-1
learn.fit_one_cycle(4,slice(lr1,lr2))
# lr1 = 1e-3
lr = 1e-1
learn.fit_one_cycle(20,slice(lr))
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,slice(1e-4,1e-3))
learn.recorder.plot_losses()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
img = open_image('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0001-0001.jpeg')
print(learn.predict(img)[0])
learn.export(file = Path("/kaggle/working/export.pkl"))
learn.model_dir = "/kaggle/working"
learn.save("stage-1",return_path=True)
from IPython.display import Image
img = open_image('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0001-0001.jpeg')
print(learn.predict(img)[0])
Image(filename="../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0001-0001.jpeg", width= 500, height=600)
Img = open_image('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg')
print(learn.predict(Img)[0])
Image(filename="../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg", width= 500, height=600)