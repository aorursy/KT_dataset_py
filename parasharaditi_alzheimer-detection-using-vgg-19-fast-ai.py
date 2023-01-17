import numpy as np
import pandas as pd
import os
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
img = open_image(Path('//kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/MildDemented/mildDem107.jpg'))
print(img.shape)
img
img = open_image(Path('//kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/VeryMildDemented/verymildDem1005.jpg'))
print(img.shape)
img
img = open_image(Path('//kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/ModerateDemented/moderateDem17.jpg'))
print(img.shape)
img
img = open_image(Path('//kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/NonDemented/nonDem1.jpg'))
print(img.shape)
img


PATH = Path('/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/')


transform = get_transforms(max_rotate=7.5, max_zoom=1.15, max_lighting=0.15, max_warp=0.15, p_affine=0.8, p_lighting = 0.8, 
                           xtra_tfms= [pad(mode='zeros'), symmetric_warp(magnitude=(-0.1,0.1)), cutout(n_holes=(1,3), length=(5,5))])
data = ImageDataBunch.from_folder(PATH, train="train/",
                                  test="test/",
                                  valid_pct=.4,
                                  ds_tfms=transform,
                                  size=112,bs=32, 
                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(10,10))

Category.__eq__ = lambda self, that: self.data == that.data
Category.__hash__ = lambda self: hash(self.obj)
Counter(data.train_ds.y)
learn = cnn_learner(data, models.vgg19_bn, metrics=accuracy, wd=1e-1)#,pretrained=False)
learn.fit_one_cycle(10)
