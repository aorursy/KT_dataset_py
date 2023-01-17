import numpy as np

import pandas as pd

import pathlib

import os

import gc



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *

DATA_DIR='../input/frutas/classes'
os.listdir(f'{DATA_DIR}')
torch.cuda.is_available()
data = ImageDataBunch.from_folder(DATA_DIR, train=".", 

                                  valid_pct=0.2,

                                  size=224, bs=10, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))
learnR50 = create_cnn(data, models.resnet50, metrics=accuracy, model_dir="/tmp/model/")

learnV16 = create_cnn(data, models.vgg16_bn, metrics=accuracy, model_dir="/tmp/model/")

learnALX = create_cnn(data, models.alexnet, metrics=accuracy, model_dir="/tmp/model/")
learnR50.fit_one_cycle(5)

learnV16.fit_one_cycle(5)

learnALX.fit_one_cycle(5)
interpR = ClassificationInterpretation.from_learner(learnR50)

interpV = ClassificationInterpretation.from_learner(learnV16)

interpA = ClassificationInterpretation.from_learner(learnALX)
interpR.plot_top_losses(9, figsize=(15,11))

interpV.plot_top_losses(9, figsize=(15,11))

interpA.plot_top_losses(9, figsize=(15,11))
interpR.plot_confusion_matrix(figsize=(8,8), dpi=60)

interpV.plot_confusion_matrix(figsize=(8,8), dpi=60)

interpA.plot_confusion_matrix(figsize=(8,8), dpi=60)