# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!pip install torchtext==0.2.3
!pip install fastai==0.7.0
PATH = "/kaggle/working/"
DATA_PATH = "../input/"
sz=224
!rm -rf {PATH}/models
!rm -rf {PATH}/tmp
os.listdir(PATH)
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
label_title_map = ["mantled howler", "patas monkey", "bald uakari",  "japanese macaque",
                  "pygmy marmoset", "white headed capuchin", "silvery marmoset",
                  "common squirrel monkey", "black headed night monkey", "nilgiri langur"]
def plot_image():
    for class_label in range(10):
        fig = plt.figure(figsize=(50,50))
        files = os.listdir(f'{DATA_PATH}validation/validation/n{class_label}')[:4]
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for i in range(1, 4):
            img = plt.imread(f'{DATA_PATH}validation/validation/n{class_label}/{files[i-1]}')
            ax = fig.add_subplot(10,3,i)
            ax.imshow(img)
            ax.set_title(label_title_map[class_label], fontsize=50)
plot_image()
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz), trn_name='../input/training/training', val_name='../input/validation/validation')
learn = ConvLearner.pretrained(arch, data, precompute=False)
lrf=learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
data.val_y
data.classes
learn = ConvLearner.pretrained(arch, data, precompute=False)
learn.fit(0.01, 3)
log_preds = learn.predict()
log_preds.shape
log_preds[:10]
preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[1,:])
preds
def rand_by_mask(mask, sample): 
    return np.random.choice(np.where(mask)[0], min(len(preds), sample), replace=False)
def rand_by_correct(is_correct, sample=3): return rand_by_mask((preds == data.val_y)==is_correct, sample)
def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    f.subplots_adjust(hspace=1, wspace=1)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = ['Prediction: %s \n Actual: %s'%(label_title_map[preds[x]],label_title_map[data.val_y[x]]) for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8)) if len(imgs)>0 else print('Not Found.')
plot_val_with_title(rand_by_correct(True, 3), "Correctly classified")
plot_val_with_title(rand_by_correct(False, 1), "Incorrectly classified")
