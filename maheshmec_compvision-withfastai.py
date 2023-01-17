%reload_ext autoreload
%autoreload 2
%matplotlib inline

from albumentations import *
import cv2

import os
print(os.listdir("../input"))

#!pip install pretrainedmodels
from tqdm import tqdm_notebook as tqdm
from torchvision.models import *
#import pretrainedmodels

from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
from fastai.callbacks import * 

#from utils import *
import sys
path = "../input/emergency-vs-nonemergency-vehicle-classification/dataset"
import numpy as np
import pandas as pd
#trainAV = pd.read_csv("../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/train.csv")
trainAV = pd.read_csv("../input/cleantrain/clean_tr.csv")
testAV = pd.read_csv("../input/emergency-vs-nonemergency-vehicle-classification/dataset/test_vc2kHdQ.csv")
trainAV.shape
trainAV.info()
trainAV.head()
tfms = get_transforms(max_rotate=90.0, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                      p_affine=1.0, p_lighting=1.)
bs = 32 #with image size 299, bs=48 & above will allocate more memory
sz = 299
np.random.seed(42)
data = ImageDataBunch.from_csv('../input', folder = 'emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images', csv_labels = 'cleantrain/clean_tr.csv',
                               valid_pct=0.10,size = sz, ds_tfms = tfms,bs=bs)
data.normalize(imagenet_stats)
data.show_batch(rows = 3)
data.train_ds
def _plot(i,j,ax):
    x,y = data.train_ds[3]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,8))
print(data.classes); data.c
gc.collect()
learnResnet34 = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True)
learnResnet34.fit_one_cycle(1)
gc.collect()
learnResnet50 = cnn_learner(data, models.resnet50, metrics=error_rate, bn_final=True)
learnResnet50.fit_one_cycle(1)
gc.collect()
learnResnet152 = cnn_learner(data, models.resnet152, metrics=accuracy, bn_final=True)#error_rate
learnResnet152.fit_one_cycle(1)
learnResnet152.model_dir = "/kaggle/working/models"
learnResnet152.save("stage-152-1")
learnResnet152.unfreeze()
learnResnet152.fit_one_cycle(1)
learnResnet152.model_dir = "/kaggle/working/models"
learnResnet152.save("stage-152-2")
learnResnet152.load('stage-152-2');
learnResnet152.lr_find()

learnResnet152.recorder.plot()
learnResnet152.load('stage-152-2');
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.save('stage-152-fin1');
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(1, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(1, max_lr=slice(3e-7,1e-3))
learnResnet152.load('stage-152-fin1');
learnResnet152.freeze()
learnResnet152.fit_one_cycle(3, max_lr=slice(1e-3))
learnResnet152.fit_one_cycle(3, max_lr=slice(1e-3))
learnResnet152.fit_one_cycle(3, max_lr=slice(3e-7,1e-3))
learnResnet152.fit_one_cycle(1, max_lr=slice(1e-3))
learnResnet152.unfreeze()
learnResnet152.fit_one_cycle(3, max_lr=slice(1e-4))
learnResnet152.fit_one_cycle(1, max_lr=slice(1e-4))
learnResnet152.fit_one_cycle(1, max_lr=slice(1e-5))
learnResnet152.fit_one_cycle(1, max_lr=slice(1e-5))
learnResnet152.fit_one_cycle(1, max_lr=slice(1e-5))
learnResnet152.fit_one_cycle(5, max_lr=slice(1e-5))
learnResnet152.load('stage-152-fin1');
interp = ClassificationInterpretation.from_learner(learnResnet152)
losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(4, figsize=(15,11))
interp.plot_confusion_matrix()
learnVGG = cnn_learner(data, models.vgg19_bn, metrics=accuracy, bn_final=True)#error_rate
#learnVGG =  VGG16()
#ConvLearner.pretrained
learnVGG.fit_one_cycle(3)
learnVGG.model_dir = "/kaggle/working/models"
learnVGG.save("stage-vgg-1")
learnVGG.unfreeze()
learnVGG.fit_one_cycle(1)
learnVGG.save("stage-vgg-2")
learnVGG.lr_find()
learnVGG.recorder.plot()
learnVGG.fit_one_cycle(6,slice(1e-6,1e-5))

interpvg = ClassificationInterpretation.from_learner(learnVGG)
losses,idxs = interpvg.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interpvg.plot_top_losses(9, figsize=(15,11))
interpvg.plot_confusion_matrix()
dft = pd.read_csv('../input/emergency-vs-nonemergency-vehicle-classification/dataset/test_vc2kHdQ.csv')
dft.head()
dft.shape
dt_test = ImageList.from_df(dft, '../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images')
# To get image from images folder based on name from test.csv
# str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][1])
img = open_image(str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][1]))
pred_class,pred_idx,outputs = learnResnet152.predict(img)
pred_class

defaults.device = torch.device('cpu')
labl =[]
for i in range(dft.shape[0]): #
    img = open_image(str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][i]))
    #img = img.normalize(imagenet_stats)
    pred_class,pred_idx,outputs = learnResnet152.predict(img)
    labl.append(pred_class)
defaults.device = torch.device('cpu')
labl =[]
for i in range(dft.shape[0]): #
    img = open_image(str('../input/emergency-vs-nonemergency-vehicle-classification/dataset/train_SOaYf6m/images/'+dft["image_names"][i]))
    #img = img.normalize(imagenet_stats)
    pred_class,pred_idx,outputs = learnVGG.predict(img)
    labl.append(pred_class)
len(labl)
sample = pd.read_csv('../input/emergency-vs-nonemergency-vehicle-classification/dataset/sample_submission_yxjOnvz.csv')
sample.head()
#create datafarme for submission & export
sample['image_names'] = testAV['image_names']
sample['emergency_or_not'] = labl
sample.to_csv('submit141.csv', index=False)