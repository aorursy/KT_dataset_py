import numpy as np

import pandas as pd

import pathlib



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *
DATA_DIR='../input/dataset'
os.listdir(f'{DATA_DIR}')
torch.cuda.is_available()
data = ImageDataBunch.from_folder(DATA_DIR, train=".",

                                  valid_pct=0.2,

                                  size=224, bs=10,

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))
learn1 = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/")

learn2 = cnn_learner(data, models.squeezenet1_0, metrics=accuracy, model_dir="/tmp/model/")

learn3 = cnn_learner(data, models.densenet201, metrics=accuracy, model_dir="/tmp/model/")
learn1.fit_one_cycle(5)

learn2.fit_one_cycle(5)

learn3.fit_one_cycle(5)
interp1 = ClassificationInterpretation.from_learner(learn1)

interp2 = ClassificationInterpretation.from_learner(learn2)

interp3 = ClassificationInterpretation.from_learner(learn3)
interp1.plot_top_losses(15, figsize=(15,11))

interp2.plot_top_losses(15, figsize=(15,11))

interp3.plot_top_losses(15, figsize=(15,11))
interp1.plot_confusion_matrix(figsize=(8,8), dpi=60)

interp2.plot_confusion_matrix(figsize=(8,8), dpi=60)

interp3.plot_confusion_matrix(figsize=(8,8), dpi=60)