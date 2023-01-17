import numpy as np

import pandas as pd

import pathlib



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *



!pip install torchsummary

from torchsummary import summary
DATA_DIR='../input/fruits/fruits'
os.listdir(f'{DATA_DIR}')
channels = 3

IMG_SIZE = 224
data = ImageDataBunch.from_folder(DATA_DIR, train=".", 

                                  valid_pct=0.2,

                                  size=IMG_SIZE, bs=10, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
# Show model layout

summary(learn.model, input_size=(channels, IMG_SIZE, IMG_SIZE)) 
learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(12,12))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)