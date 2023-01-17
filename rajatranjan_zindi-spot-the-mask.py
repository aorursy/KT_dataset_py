# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pathlib import Path

from fastai import *

from fastai.vision import *

import torch

from fastai.callbacks.hooks import *

from tqdm import tqdm_notebook



import seaborn as sns

import matplotlib.pyplot as plt

import os

import cv2

import glob
train=pd.read_csv('/kaggle/input/spot-the-mask/train_labels.csv')

s=pd.read_csv('/kaggle/input/spot-the-mask/sample_sub.csv')
train
s.rename(columns={'+ACI-image+ACI-':'image','+ACI-target+ACI-':'target'},inplace=True)

s

s['image']=s['image'].apply(lambda x: x[5:-5])
# data_folder = Path("../input/hackerearth-dl-challengeautotag-images-of-gala/dataset")

data_path = "/kaggle/input/spot-the-mask/images"

# path = os.path.join(data_path , "*jpg")

# path
train['target'].value_counts()
##transformations to be done to images



tfms = get_transforms(do_flip=True,flip_vert=False 

                      ,max_rotate=12.0, max_zoom=1.6, max_lighting=0.5, max_warp=0.1, p_affine=0.9,

                      p_lighting=0.55

                     )

#, xtra_tfms=zoom_crop(scale=(0.9,1.8), do_rand=True, p=0.8))



## create databunch of test set to be passed

test_img = ImageList.from_df(s, path=data_path, folder='')

test_img
np.random.seed(145)

## create source of train image databunch

src = (ImageList.from_df(train, path=data_path, folder='')

       .split_by_rand_pct(0.10)

       #.split_none()

       .label_from_df()

       .add_test(test_img))



src
data = (src.transform(tfms, size=299,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)

        .databunch(path='.', bs=32, device= torch.device('cuda:0')).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,12))
print(data.classes)
from fastai.metrics import error_rate # 1 - accuracy

np.random.seed(42)

learn = cnn_learner(data=data, base_arch=models.resnet101, metrics=[error_rate, accuracy],

                    callback_fns=ShowGraph)
#lets find the correct learning rate to be used from lr finder

np.random.seed(42)

learn.freeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
slice(3.0E-0,3.0E-02,0.005)
# lr = 5.79E-04

np.random.seed(42)

#learn.fit_one_cycle(10, slice(lr))

learn.fit_one_cycle(10, slice(3.0E-03,3.0E-02), wd=0.25)

learn.save('spotmask-stage-1')
#lets plot the lr finder record

# np.random.seed(42)

# learn.unfreeze()

# learn.lr_find()



# learn.recorder.plot(suggestion=True)
# np.random.seed(42)

# learn.fit_one_cycle(5,slice(6.31E-07),wd=0.25)

# learn.save('spotmask-stage-2')
# learn.freeze_to(-1)

# learn.lr_find()



# learn.recorder.plot(suggestion=True)
# learn.fit_one_cycle(5,slice(6.31E-07,1.0E-06),wd=0.1)

# learn.save('spotmask-stage-3')



# learn.unfreeze()

# learn.lr_find()



# learn.recorder.plot(suggestion=True)
# learn.freeze_to(-2)

# learn.lr_find()



# learn.recorder.plot(suggestion=True)
# learn.fit_one_cycle(3,slice(7.59E-07),wd=0.25)

# learn.save('spotmask-stage-4')
# learn.freeze_to(-1)

# learn.lr_find()



# learn.recorder.plot(suggestion=True)
# learn.fit_one_cycle(7,slice(7.59E-07),wd=0.25)

# learn.save('spotmask-stage-4')
#lets see the most mis-classified images (on validation set)

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,10))
##learn.TTA improves score further. lets see for the validation set

pred_val,y = learn.TTA(ds_type=DatasetType.Valid)

len(pred_val)
# np.array(pred_val)[:,-1]
# [pred_val[i] for i in range(len(pred_val))]

y
def logloss(true_label, predicted, eps=1e-15):

    p = np.clip(predicted, eps, 1 - eps)

    if true_label == 1:

        return -log(p)

    else:

        return -log(1 - p)
np.argmax(pred_val,1)
from sklearn.metrics import f1_score, accuracy_score,log_loss

# valid_preds = [np.argmax(pred_val[i]) for i in range(len(pred_val))]

# valid_preds = np.array(valid_preds)

# valid_preds = np.array(pred_val)[:,-1]

valid_preds = np.argmax(pred_val,1)

# y = np.array(y+1)

log_loss(valid_preds,y)
valid_preds
preds,_ = learn.TTA(ds_type=DatasetType.Test)

#preds,_ = learn.get_preds(ds_type = DatasetType.Test)

labelled_preds = np.array(preds)[:,-1]



labelled_preds
labelled_preds
df = pd.DataFrame({'image':s['image'], 'target':labelled_preds})

df
df.to_csv('submission9.csv', index=False)