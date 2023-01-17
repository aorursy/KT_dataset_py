import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline 

import cv2

import os
os.listdir("../input")
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=224
fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])
labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])
arch=resnet50
data = ImageClassifierData.from_names_and_array(
    path=PATH, 
    fnames=fnames, 
    y=labels, 
    classes=['dogs', 'cats'], 
    test_name='test', 
    tfms=tfms_from_model(arch, sz)
)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
###
### Search for suitable, i.e., best Learning Rate for our-newly-added-Last Layer (as we have used 'precompute=True', i.e., ResNet50-minus-its-last-layer weights are being re-used as is)
###
#lrf=learn.lr_find()
#learn.sched.plot_lr()
#learn.sched.plot()

###
### Use the identified best Learning Rate for our-newly-added-Last Layer
### Note that even without running above 3 lines of Learning Rate Finder, it is well known that best learning rate is 0.01 for Cats & Dogs images with 224x224 size
### Kaggle Score obtained is 0.38683 (v7)
###
#learn.fit(0.01, 2)
###
### SGDR (SGD with warm Resrart): fast.ai uses half Cosine shape decay (start with 0.01 & decay till 0) of LR during each epoch and then it restarts with 1e-02
### Kaggle score obtained is 0.37578 (v8)
###
learn.fit(1e-2, 10, cycle_len=1)
learn.sched.plot_lr()
###
### Continue from Last Layer learned model with PreCompute=TRUE
### Unfreeze all layers (all weights learned so far are retained) => it sets PreCompute=FALSE making all layers learnable
### Effectively, the network weights are intialized as (ResNet-minus-last-layer with its original pre-trained weight & Last Layer as per above model learning while keeping ResNet as frozen)
### Now, all layers are FURTHER learnable
### Kaggle score obtained is 0.34815 (v9)
###
learn.unfreeze()

# Differential LR (above identified best LR for last layer, x0.1 to middle layer, x0.01 to inner layer)
lr=np.array([1e-4,1e-3,1e-2])

learn.fit(lr, 3, cycle_len=1, cycle_mult=2)

learn.sched.plot_lr()
temp = learn.predict(is_test=True)
pred = np.argmax(temp, axis=1)
import cv2

# learn.predict works on unsorted os.listdir, hence listing filenames without sorting
fnames_test = np.array([f'test/{f}' for f in os.listdir(f'{PATH}test')])

f, ax = plt.subplots(5, 5, figsize = (15, 15))

for i in range(0,25):
    imgBGR = cv2.imread(f'{PATH}{fnames_test[i]}')
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    
    # a if condition else b
    predicted_class = "Dog" if pred[i] else "Cat"

    ax[i//5, i%5].imshow(imgRGB)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(predicted_class))    

plt.show()
results_df = pd.DataFrame(
    {
        'id': pd.Series(fnames_test), 
        'label': pd.Series(pred)
    })
results_df['id'] = results_df.id.str.extract('(\d+)')
results_df['id'] = pd.to_numeric(results_df['id'], errors = 'coerce')
results_df.sort_values(by='id', inplace = True)

results_df.to_csv('submission.csv', index=False)
results_df.head()