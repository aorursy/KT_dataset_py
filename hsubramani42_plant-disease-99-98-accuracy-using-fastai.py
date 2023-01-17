from fastai import *
from fastai.vision import *
data=ImageDataBunch.from_folder('../input/plantdisease',valid_pct=0.20,ds_tfms=get_transforms(),size=224).normalize(imagenet_stats)
model=cnn_learner(data,models.densenet201,metrics=accuracy)
model.fit_one_cycle(6)
