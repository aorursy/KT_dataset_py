%reload_ext autoreload

%autoreload 2

%matplotlib inline

# The notebook reloads automatically, have plots stored in document
from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate



import pandas as pd # Data processing

import numpy as np



import matplotlib.pyplot as plt

from pathlib import Path

# Import some libraries to use
data_folder = Path("../input/aptos2019-blindness-detection")

data_folder.ls()

# Allows for the manipulation of folders, see file structure used
train_df = pd.read_csv(data_folder/'train.csv')

test_df = pd.read_csv(data_folder/'test.csv')

# Read data into the dataframes for training and testing, allows for easy manipulation of the data
test_data = ImageList.from_df(test_df, path=data_folder, folder='test_images', suffix='.png')

# Object to represent test set, using data block API
bs = 64

#Set batch size
# Data block API, prepares data for training, use in the model 

data = (ImageList.from_df(train_df, path=data_folder, folder='train_images', suffix = '.png')

                 .split_by_rand_pct(valid_pct=0.2) #split between training, validation sets

                 .label_from_df()

                 .add_test(test_data)              

                 .transform(get_transforms(flip_vert=True), size=224)

                 .databunch(path='.', bs=bs) #use bs from earlier

                 .normalize(imagenet_stats)

       )
data.show_batch(rows=3, figsize=(10,10))

# See some of the images we are working with
# Create the learner for a convolutional neural network

learn = cnn_learner(data, models.resnet34, pretrained=False, model_dir = "/output/kaggle/working", metrics=error_rate)
# Find optimal learning rate to use for the model

learn.lr_find()

learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
learn.fit_one_cycle(10, min_grad_lr)
#Retrain model and fine-tune the learning rate

learn.save('initial') # Set of weights created with learn, saves the model for possible future use/access

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
# Fine-tuning, new learning rate

min_grad_lr2 = learn.recorder.min_grad_lr

min_grad_lr2
learn.fit_one_cycle(10, min_grad_lr2)

learn.save('final')