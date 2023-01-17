%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pandas as pd

import numpy as np

import os

import matplotlib as plt

import sklearn as sk



import fastai

from fastai.vision import *

from fastai.callbacks import *

from fastai.utils.mem import *

from torchvision.models import resnet50
path = untar_data(URLs.PETS); path
help(untar_data)
from torchvision.models import resnet18 #First train on resnet 18 then improve results using resnet50. 
#Folders in file.

path.ls()
anot_path = path/"annotations"

img_path = path/"images"

print("Annotation Path: {}".format(anot_path))

print("Image Path: {}".format(img_path))

file_names = get_image_files(img_path) #get_image_files function gives array of image files in the folder.

file_names[:5]
np.random.seed(2)

pat = re.compile(r'/([^/]+)_\d+.jpg$')
data = ImageDataBunch.from_name_re(img_path, file_names, pat, ds_tfms=get_transforms(), size=224, bs=16, num_workers=0

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
print(data.valid_ds.databunch)

print(data.train_ds.databunch)
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
preds,y,losses = learn.get_preds(with_loss=True)

predictions = [np.argmax(x).item() for x in preds]

y_actual = [i.item() for i in y]

print(predictions[:5])

print(y_actual[:5])

print(losses[:5])
interp = ClassificationInterpretation.from_learner(learn)

#interp = ClassificationInterpretation(learn, preds, y, losses)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
#doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#interp.confusion_matrix()
#interp.most_confused(min_val=2)
#classification_report

from sklearn.metrics import classification_report

print(classification_report(y_actual, predictions , target_names=data.classes))



learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)
learn = create_cnn(data, models.resnet101,  metrics=error_rate)
learn.fit_one_cycle(6)
learn.save('stage_resnet101_-1')
preds,y,losses = learn.get_preds(with_loss=True)
predictions = [np.argmax(x).item() for x in preds]

y_actual = [i.item() for i in y]

print(predictions[:5])

print(y_actual[:5])

print(losses[:5])
interp = ClassificationInterpretation.from_learner(learn)

#interp = ClassificationInterpretation(learn, preds, y, losses)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#classification_report

from sklearn.metrics import classification_report

print(classification_report(y_actual, predictions , target_names=data.classes))