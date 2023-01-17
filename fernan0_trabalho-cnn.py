import numpy as np

import pandas as pd

import pathlib



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *
DATA_DIR='/kaggle/input/transfers/trabalho transfer/root'
os.listdir(f'{DATA_DIR}')
torch.cuda.is_available()
data = ImageDataBunch.from_folder(DATA_DIR, train=".", 

                                  valid_pct=0.2,

                                  size=224, bs=10, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/model/")
learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)