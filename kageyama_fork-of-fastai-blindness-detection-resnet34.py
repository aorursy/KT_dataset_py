# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image

from sklearn.utils import shuffle



print(os.listdir("../input"))
# copy pretrained weights for resnet34 to the folder fastai will search by default

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'
!mkdir ../data

!mkdir ../data/train
!mkdir ../data/train/0

!mkdir ../data/train/1

!mkdir ../data/train/2

!mkdir ../data/train/3

!mkdir ../data/train/4
print(os.listdir("../data/train"))
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



x_train = df_train['id_code']

y_train = df_train['diagnosis']
import subprocess

def move_img(x,y,kind):

    for id_code ,diagnosis in tqdm(zip(x,y)):

        if diagnosis == 0:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/0/{}.png'.format(kind,id_code)])

        if diagnosis == 1:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/1/{}.png'.format(kind,id_code)])

        if diagnosis == 2:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/2/{}.png'.format(kind,id_code)])

        if diagnosis == 3:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/3/{}.png'.format(kind,id_code)])

        if diagnosis == 4:

            subprocess.call(['cp','../input/aptos2019-blindness-detection/{}_images/{}.png'.format(kind,id_code),'../data/{}/4/{}.png'.format(kind,id_code)])
move_img(x_train,y_train,'train')
print(os.listdir("../data/train/")) 
# create image data bunch

data = ImageDataBunch.from_folder('../data/', 

                                  train="../data/train", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,

                                  bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
# check classes

print(f'Classes: \n {data.classes}')
# show some sample images

data.show_batch(rows=3, figsize=(7,6))
# build model (use resnet34)

learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
# first time learning

learn.fit_one_cycle(6,1e-2)
# save stage

learn.save('stage-1')
# search appropriate learning rate

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
# second time learning

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-5 ))
# save stage

learn.save('stage-2')
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)