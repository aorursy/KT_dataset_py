%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline 

import cv2
import os

from keras.datasets import mnist

from fastai.conv_learner import *
from fastai.plots import *

from sklearn.model_selection import train_test_split
os.listdir("../input")
# 42k train & 28k test images of size 28x28 are available in row-per-image flattened csv
train_img_lbl_kaggle = pd.read_csv("../input/train.csv")
test_img_kaggle = pd.read_csv("../input/test.csv")
# train's 1st column is label
train_img_kaggle = train_img_lbl_kaggle.iloc[:, 1:]
train_lbl_kaggle = train_img_lbl_kaggle.iloc[:, 0:1]
train_img_kaggle = train_img_kaggle.values.reshape(-1, 28, 28)
test_img_kaggle = test_img_kaggle.values.reshape(-1, 28, 28)

(train_img_kaggle.shape, test_img_kaggle.shape)
(train_img_keras, train_lbl_keras), (test_img_keras, test_lbl_keras) = mnist.load_data()
train_lbl_keras = train_lbl_keras.reshape(60000, 1)
(train_lbl_kaggle.shape, train_lbl_keras.shape)
train_img_all = np.concatenate((train_img_kaggle, train_img_keras), axis=0)
train_lbl_all = np.concatenate((train_lbl_kaggle, train_lbl_keras), axis=0)
(train_img_all.shape, train_lbl_all.shape)
# Converting images from 8-bit to 24-bit 
train_img_all = np.stack((train_img_all,)*3, axis = -1).astype('float32')
test_img_kaggle = np.stack((test_img_kaggle,)*3, axis = -1).astype('float32')

(train_img_all.shape, test_img_kaggle.shape)
train_img, val_img, train_lbl, val_lbl = train_test_split(train_img_all, train_lbl_all, train_size=0.8, random_state=1, stratify=train_lbl_all)
train_lbl = train_lbl.flatten()
val_lbl = val_lbl.flatten()
(train_lbl.shape, val_lbl.shape)
# Though 30' random rotation loos quite large, it gave good results with limited samples
# This relatively large random roation was tried to check if it helps avoid mis-labeling

arch = resnet50
sz = 28
classes = np.unique(train_lbl)
data = ImageClassifierData.from_arrays(path = "/tmp",
                                     trn = (train_img, train_lbl),
                                     val = (val_img, val_lbl),
                                     classes = train_lbl,
                                     test = test_img_kaggle,
                                     tfms = tfms_from_model(arch, sz, aug_tfms = [RandomRotateZoom(deg=10, zoom=1.1, stretch=1.0)]))
learn = ConvLearner.pretrained(arch, data, precompute = True)
###
### Search for suitable, i.e., best Learning Rate for our-newly-added-Last Layer (as we have used 'precompute=True', i.e., ResNet50-minus-its-last-layer weights are being re-used as is)
###
#lrf=learn.lr_find()
#learn.sched.plot_lr()

#learn.sched.plot()

###
### Use the identified best Learning Rate for our-newly-added-Last Layer
### Note that even without running above 3 lines of Learning Rate Finder, it is well known that best learning rate is 0.01 even for MNIST Digits 28x28 images
###
#learn.fit(0.01, 2)
###
### SGDR (SGD with warm Resrart): fast.ai uses half Cosine shape decay (start with 0.01 & decay till 0) of LR during each epoch and then it restarts with 1e-02
###
learn.fit(1e-2, 10, cycle_len = 1)
learn.sched.plot_lr()
###
### Continue from Last Layer learned model with PreCompute=TRUE
### Unfreeze all layers (all weights learned so far are retained) => it sets PreCompute=FALSE making all layers learnable
### Effectively, the network weights are intialized as (ResNet-minus-last-layer with its original pre-trained weight & Last Layer as per above model learning while keeping ResNet as frozen)
### Now, all layers are FURTHER learnable
###
learn.unfreeze()

# Differential LR (above identified best LR for last layer, x0.1 to middle layer, x0.01 to inner layer)
lr=np.array([1e-4, 1e-3, 1e-2])

learn.fit(lr, 3, cycle_len = 1, cycle_mult =  2)
learn.sched.plot_lr()
#temp = learn.predict(is_test = True)
#pred = np.argmax(temp, axis = 1)

log_preds, y = learn.TTA(is_test=True)
probs_test = np.mean(np.exp(log_preds), 0)

pred_df = pd.DataFrame(probs_test)
pred_df = pred_df.assign(Label = pred_df.values.argmax(axis=1))
pred_df = pred_df.assign(ImageId = pred_df.index.values + 1)
submit_df = pred_df[['ImageId', 'Label']]
submit_df.shape
f, ax = plt.subplots(5, 5, figsize = (15, 15))

for i in range(0,25):
    ax[i//5, i%5].imshow(test_img_kaggle[i].astype('int'))
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(submit_df.Label[i]))    

plt.show()
submit_df.to_csv('submission.csv', index=False)