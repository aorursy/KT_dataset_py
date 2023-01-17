import pandas as pd
gb = pd.read_csv("../input/garbage/spotgarbage-GINI-master/spotgarbage/garbage-queried-images.csv")

no_gb = pd.read_csv("../input/garbage/spotgarbage-GINI-master/spotgarbage/non-garbage-queried-images.csv")
gb = gb.drop(['startX','startY','endX','endY','query'],axis=1)

no_gb = no_gb.drop(['query'],axis=1)
gb.head()
no_gb.head()
no_gb["label"] = 0.0
gb["label"].isna().sum()
PATH = "../input/garbage/spotgarbage-GINI-master/spotgarbage"
import glob

import cv2

import numpy as np

import tensorflow as tf

import os

import pandas as pd
print(gb.shape)

print(no_gb.shape)
cols = ["path","non-garbage-queried-images" ,"garbage-queried-images"]
data = pd.DataFrame(columns = cols, index= range(0,2512))
data[cols] = 0

data.head()
for x in os.walk(PATH):

    directory = x[1]

    break

print(directory)
c = 0

for dir in directory:

    print(PATH +dir)

    for i in glob.glob(PATH+ "/" +dir +'/*', recursive=True):

        data.loc[c,"path"] = i

        data.loc[c,dir] =1

        c = c + 1





print(str(c))
data.head()
data[data["ambiguous-annotated-images"] == 1.0].shape
df = data[data["ambiguous-annotated-images"] != 1.0]
df.shape
df = df.drop(["ambiguous-annotated-images"], axis=1)
df.head()
from fastai import *

from fastai.vision import *

from torch import  *

from fastai.callbacks import *

import warnings

import torch

import torchvision.models as torch_models

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
!mkdir "garbage-queried-images"

!mkdir "non-garbage-queried-images"
PATH = "/kaggle/working"
!cp "../input/garbage/spotgarbage-GINI-master/spotgarbage/garbage-queried-images/"*/* "/kaggle/working/garbage-queried-images/"
!cp "../input/garbage/spotgarbage-GINI-master/spotgarbage/non-garbage-queried-images/"*/* "/kaggle/working/non-garbage-queried-images/"
!ls "/kaggle/working/garbage-queried-images/" | wc -l
!ls "/kaggle/working/non-garbage-queried-images/" | wc -l
np.random.seed(42)

data=ImageDataBunch.from_folder(PATH,train="train",

                                ds_tfms=get_transforms(do_flip=True,flip_vert=True),valid_pct=0.1,size=350,num_workers=0,bs=32).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
classes=["non-garbage-queried-images","garbage-queried-images"]

for c in classes :

    print(c)

    verify_images (PATH + c, max_workers=2)
learn= cnn_learner(data,torch_models.wide_resnet50_2, metrics=[accuracy,Recall(),error_rate])
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-4,1e-3),callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])
interpreter = ClassificationInterpretation.from_learner(learn)

interpreter.most_confused(min_val=11)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6),callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])
learn.save('model')
learn.export()