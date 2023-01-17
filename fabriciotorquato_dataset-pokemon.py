import numpy as np

import pandas as pd

import pathlib



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *
DATA_DIR='../input/Pokemon'
os.listdir(DATA_DIR)
torch.cuda.is_available()
data = ImageDataBunch.from_folder(DATA_DIR, train=".", 

                                  valid_pct=0.2,

                                  size=224, bs=10, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))
resnet18 = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/")

alexnet = cnn_learner(data, models.alexnet, metrics=accuracy, model_dir="/tmp/model/")

densenet201 = cnn_learner(data, models.densenet201, metrics=accuracy, model_dir="/tmp/model/")
resnet18.fit_one_cycle(10)

alexnet.fit_one_cycle(10)

densenet201.fit_one_cycle(10)
irnterpretation_resnet18 = ClassificationInterpretation.from_learner(resnet18)

irnterpretation_alexnet = ClassificationInterpretation.from_learner(alexnet)

irnterpretation_densenet201 = ClassificationInterpretation.from_learner(densenet201)
irnterpretation_resnet18.plot_top_losses(9, figsize=(10,10))
irnterpretation_alexnet.plot_top_losses(9, figsize=(10,10))
irnterpretation_densenet201.plot_top_losses(9, figsize=(10,10))
irnterpretation_resnet18.plot_confusion_matrix(figsize=(4,4), dpi=60)

irnterpretation_alexnet.plot_confusion_matrix(figsize=(4,4), dpi=60)

irnterpretation_densenet201.plot_confusion_matrix(figsize=(4,4), dpi=60)