# Put these at the top of every notebook, to get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

from subprocess import check_output
print(check_output(["ls", "../input/dr_data/DR_data/"]).decode("utf8"))
PATH ="../input/dr_data/DR_data/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
arch=resnet34
sz=224
tfms=tfms_from_model(arch, sz)

torch.cuda.is_available()
torch.backends.cudnn.enabled
fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])

img = plt.imread(f'{PATH}{fnames[0]}')
plt.imshow(img);
img.shape
img[2000:2004,1500:1504]
# Uncomment the below if you need to reset your precomputed activations
# shutil.rmtree(f'{PATH}tmp', ignore_errors=True)
data= ImageClassifierData.from_csv(path=PATH,
                                      folder='train', 
                                      csv_fname='../input/trainLabels_3.csv'
                                      , tfms=tfms, test_name='test', 
                                       suffix='.jpeg')
# Uncomment the below for training
'''
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.01, 2)
'''
data
# This is the label for a val data
data.val_y
# from here we know that 'cats' is label 0 and 'dogs' is label 1.
data.classes
# this gives prediction for validation set. Predictions are in log scale
#uncomment below
#log_preds = learn.predict()
#log_preds.shape
#log_preds[:10]
preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1
probs = np.exp(log_preds[:,0])        # pr(no DR)
def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)
def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))
# 1. A few correct labels at random
plot_val_with_title(rand_by_correct(True), "Correctly classified")
# 2. A few incorrect labels at random
plot_val_with_title(rand_by_correct(False), "Incorrectly classified")
def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)
plot_val_with_title(most_by_correct(0, True), "Most correct No Retinopathy")
plot_val_with_title(most_by_correct(4, True), "Most correct Retinopathy")
plot_val_with_title(most_by_correct(0, False), "Most incorrect No Retinopathy")
plot_val_with_title(most_by_correct(1, False), "Most incorrect Retinopathy")
most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")