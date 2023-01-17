import numpy as np 

import pandas as pd 

import seaborn as sns



from torch import nn, optim

from torchvision import transforms, models, datasets



from fastai.callbacks import *

from sklearn.metrics import roc_curve, auc

from fastai.vision import *



sns.set(style='whitegrid')

plt.style.use('seaborn-darkgrid')

import os

print(os.listdir("../input/cell_images/cell_images/"))



path = Path('../input/cell_images/cell_images/')

np.random.seed(42)



data = ImageDataBunch.from_folder(path, train='train', valid_pct=0.2, ds_tfms=get_transforms(),size=224, bs=128, num_workers=0).normalize(imagenet_stats)

data
data.classes
data.show_batch(4, figsize=(15,10))
learn = create_cnn(data, models.resnet50 , model_dir="/tmp/model/", metrics=[accuracy, error_rate])

learn
learn.fit_one_cycle(6, 1e-2, pct_start=0.05,callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])

learn.recorder.plot_losses()

plt.show()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

Learning_rate = learn.recorder.min_grad_lr

print(Learning_rate)

plt.show()
learn.fit_one_cycle(3, Learning_rate, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(4, figsize=(10,8), heatmap=False)

plt.show()
interp.plot_confusion_matrix(figsize=(10, 8))

plt.show()

interp.most_confused()
learn.show_results(ds_type=DatasetType.Valid)