# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install fastai==0.7.0

!pip install torchtext==0.2.3

!pip install torch==0.3.1
from fastai.transforms import *

from fastai.conv_learner import * 

from fastai.model import *

from fastai.dataset import *

from fastai.sgdr import *

from fastai.plots import *
from fastai.imports import *
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir("../input")
filenames = os.listdir('../input/train/train')[:5]

filenames
path = "../input/train/train"

img = plt.imread(f'{path}/{filenames[0]}')

plt.imshow(img)
img.shape
shutil.rmtree(f'{path}tmp', ignore_errors=True)#
img[:4,:4]
PATH = "../input/"

TMP_PATH = "/tmp/tmp"

MODEL_PATH = "/tmp/model/"

sz=224
fnames = np.array([ f'train/train/{i}'  for i in sorted(os.listdir(f'{PATH}train/train'))])

label = np.array([0 if 'cat' in fname else 1 for fname in fnames]).astype(np.double)
arch = resnet34

data = ImageClassifierData.from_names_and_array(path=PATH,

                                                fnames=fnames ,

                                                y=label,

                                                classes=['dogs', 'cats'],

                                                test_name=(f'{PATH}test1/test1'),

                                      tfms=tfms_from_model(arch , sz))

learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)

learn.fit(0.01 , 2)
data.val_y
data.classes
log_preds = learn.predict()

log_preds.shape
log_preds[:10]
preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1

probs = np.exp(log_preds[:,1])        # pr(dog)
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
plot_val_with_title(rand_by_correct(True), "Correctly classified")
# 2. A few incorrect labels at random

plot_val_with_title(rand_by_correct(False), "Incorrectly classified")
def most_by_mask(mask, mult):

    idxs = np.where(mask)[0]

    return idxs[np.argsort(mult * probs[idxs])[:4]]



def most_by_correct(y, is_correct): 

    mult = -1 if (y==1)==is_correct else 1

    return most_by_mask(((preds == data.val_y)==is_correct) & (data.val_y == y), mult)
plot_val_with_title(most_by_correct(0, True), "Most correct cats")
plot_val_with_title(most_by_correct(1, True), "Most correct dogs")
plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")
plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")
most_uncertain = np.argsort(np.abs(probs -0.5))[:4]

plot_val_with_title(most_uncertain, "Most uncertain predictions")
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
lrf=learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
def get_augs():

    data = ImageClassifierData.from_names_and_array(

        path=PATH, 

        fnames=fnames, 

        y=label, 

        classes=['dogs', 'cats'], 

        test_name=f'{PATH}test1/test1', 

        tfms=tfms,

        num_workers=1,

        bs=2

    )

    x,_ = next(iter(data.aug_dl))

    return data.trn_ds.denorm(x)[1]
ims = np.stack([get_augs() for i in range(6)])
plots(ims , rows=2)