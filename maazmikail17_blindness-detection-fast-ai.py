from fastai.vision import *

from fastai import *

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import cv2

import glob

import torch





device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
PATH = Path('../input/aptos2019-blindness-detection')

train = PATH/'train_images'

test = PATH/'test_images'

train_folder = 'train_images'

model_dir = Path('/kaggle/working/')

bs = 64

img_size = 512



train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))

train_df['id_code'] = train_df['id_code'].apply(lambda x: f'{train_folder}/{x}.png')

train_df.head()
PATH.ls()
sns.countplot(train_csv['diagnosis'])
print(f"Size of Training set images: {len(list(train.glob('*.png')))}")

print(f"Size of Test set images: {len(list(test.glob('*.png')))}")
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=180,

                      max_warp=0,max_zoom=1.35,p_lighting=0.4,

                      max_lighting=0.3,xtra_tfms=[flip_lr()] )


data = (ImageList.from_df(train_df, PATH)

        .split_by_rand_pct()

        .label_from_df()

        .transform(tfms, size=img_size, resize_method=ResizeMethod.PAD,padding_mode='zeros')

       ).databunch(bs=32).normalize(imagenet_stats)

data



# The more simpler ImageDataBunch shortcut method. 

# data = ImageDataBunch.from_df(PATH, train_csv, folder='train_images', 

#                               suffix='.png', no_check=True, 

#                               ds_tfms=get_transforms(), size=512, bs=32).normalize(imagenet_stats)
data.train_ds
data.valid_ds
data.show_batch(rows=3, figsize=(10,8))
# Training

kappa = KappaScore()

kappa.weights = "quadratic"



learner = cnn_learner(data, models.resnet50, metrics=[error_rate, kappa])



learner.fit_one_cycle(4)

learner.model_dir = '/kaggle/working'

learner.unfreeze()

learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(4, max_lr=slice(1e-5, 1e-3))
learner.save('stage-2', return_path = True)
tfms_456 = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,

                      max_warp=0,max_zoom=1.25,p_lighting=0.5,

                      max_lighting=0.2)



data_456 = (ImageList.from_df(train_df, PATH)

        .split_by_rand_pct()

        .label_from_df()

        .transform(tfms_456, size=456)

       ).databunch(bs=32).normalize(imagenet_stats)

data_456



learner_2 = cnn_learner(data_456, models.resnet50, metrics=[error_rate, kappa])



learner_2.load(model_dir/'stage-2')
learner_2.model_dir = '/kaggle/working'

learner_2.unfreeze()

learner_2.lr_find()
learner_2.recorder.plot()
learner_2.fit_one_cycle(4, max_lr=slice(1e-5, 1e-3))