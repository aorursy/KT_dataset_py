import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("/kaggle/input/flower-recognition/flower_recognition"))
!pip install pretrainedmodels
from torchvision.models import *

# import pretrainedmodels



from fastai import *

from fastai.vision import *

from fastai.vision.models import *

from fastai.vision.learner import model_meta

import fastai



from utils import *

import sys

import torch

fastai.__version__
torch.__version__
lis = os.listdir('/kaggle/input/flower-recognition/flower_recognition/train')
sub = pd.read_csv('/kaggle/input/flower-recognition/flower_recognition/sample_submission.csv')
sub.shape
bs=8
path = "/kaggle/input/flower-recognition/flower_recognition/train"
## test filenames to be used to create final submission.

filenames = os.listdir('/kaggle/input/flower-recognition/flower_recognition/test')
df = pd.read_csv('/kaggle/input/flower-recognition/flower_recognition/train.csv')
df.head()
# CenterCrop(32)

## These Transformation applied based upon my previous competition Experience.

## if you want to try other transformation check this link

## https://docs.fast.ai/vision.transform.html

tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0,do_flip=False,xtra_tfms=[cutout()])

tfms1 = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0,do_flip=False,xtra_tfms=[cutout()])

data = (ImageList.from_csv(path, csv_name = '../train.csv', suffix='.jpg')

        .split_by_rand_pct()              

        .label_from_df()            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms, size=400)

        .databunch(num_workers=0,bs=8))



data1 = (ImageList.from_csv(path, csv_name = '../train.csv', suffix='.jpg')

        .split_by_rand_pct()              

        .label_from_df()            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms1, size=400)

        .databunch(num_workers=0,bs=8))
## to see the images in train with there labels

data.show_batch(rows=3, figsize=(8,10))
## print the target classes

print(data.classes)
## load the pretrained imagenet model

## you can try other models from this link

## https://docs.fast.ai/vision.models.html

learn = cnn_learner(data, models.densenet169, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
## training with one cycle which used cyclic learning rate and learning rate annhelling

learn.fit_one_cycle(1)
learn.unfreeze()

learn.lr_find()
# learn.recorder.plot(suggestion=True)

# best_clf_lr = learn.recorder.min_grad_lr

# best_clf_lr
# learn.fit_one_cycle(2, max_lr=best_clf_lr)

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
## 2nd model
learn1 = cnn_learner(data1, models.densenet201, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
## training with one cycle which used cyclic learning rate and learning rate annhelling

learn1.fit_one_cycle(1)
learn1.unfreeze()

learn1.lr_find()

learn1.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
learn2 = cnn_learner(data1, models.resnet152, metrics=[error_rate, accuracy], model_dir="/tmp/model/")

## training with one cycle which used cyclic learning rate and learning rate annhelling

learn2.fit_one_cycle(1)

learn2.unfreeze()

learn2.lr_find()

learn2.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
learn3 = cnn_learner(data, models.densenet121, metrics=[error_rate, accuracy], model_dir="/tmp/model/")

## training with one cycle which used cyclic learning rate and learning rate annhelling

learn3.fit_one_cycle(1)

learn3.unfreeze()

learn3.lr_find()

learn3.fit_one_cycle(2, max_lr=slice(1e-6,1e-3))
## Applied Test Time Augmentation
preds,_ = learn.TTA(ds_type=DatasetType.Test)
preds1,_ = learn1.TTA(ds_type=DatasetType.Test)
preds2,_ = learn2.TTA(ds_type=DatasetType.Test)

preds3,_ = learn3.TTA(ds_type=DatasetType.Test)
## create the submission file 
labelled_preds = []

pred11 = preds + preds1 + preds2 + preds3

for pred in pred11:

    labelled_preds.append(int(np.argmax(pred))+1)



submission = pd.DataFrame(

    {'image_id': filenames,

     'category': labelled_preds,

    })

submission.to_csv('submission.csv',index=False)
submission.head()
submission['image_id'] = submission['image_id'].apply(lambda x:x.split('.')[0])
submission = submission.sort_values(by = ['image_id'], ascending = [True])
## To download the submission file without Commiting the kernel.
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "subm.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission)