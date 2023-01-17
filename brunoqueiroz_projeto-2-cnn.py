import numpy as np

import pandas as pd

import pathlib



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *
DATA_DIR='../input/lego-bricks-3-classes/Lego bricks'
os.listdir(f'{DATA_DIR}')
torch.cuda.is_available()
data = ImageDataBunch.from_folder(DATA_DIR, train=".", 

                                  valid_pct=0.2,

                                  size=224, bs=10, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
learn1 = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/")
learn2 = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir="/tmp/model/")
learn3 = cnn_learner(data, models.densenet169, metrics=accuracy, model_dir="/tmp/model/")
learn1.fit_one_cycle(5)
learn2.fit_one_cycle(5)
learn3.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn1)
interp2 = ClassificationInterpretation.from_learner(learn2)
interp3 = ClassificationInterpretation.from_learner(learn3)
interp.plot_top_losses(9, figsize=(20,15))
interp2.plot_top_losses(9, figsize=(20,15))
interp3.plot_top_losses(9, figsize=(20,15))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
interp2.plot_confusion_matrix(figsize=(8,8), dpi=60)
interp3.plot_confusion_matrix(figsize=(8,8), dpi=60)