%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np 
import pandas as pd 
import os
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
# here we specify  the path for our dataset and our global variables!! 
PATH = '../input/'
sz = 224 
bs = 58
arch = resnet50
print(os.listdir(PATH))
# os.makedirs('../work')
#labels.. 
labels_dir = f'{PATH}labels.csv'
n = len(list(open(labels_dir))) -1 
val_idxs = get_cv_idxs(n)
len(val_idxs)
n
labels = pd.read_csv(labels_dir)
labels.head()
labels['breed'].value_counts()
tfms = tfms_from_model(arch, sz, aug_tfms = transforms_side_on, max_zoom = 1.1)
data = ImageClassifierData.from_csv(PATH, 'train', csv_fname = labels_dir,test_name = 'test', suffix = '.jpg',val_idxs = val_idxs,tfms=tfms, bs =bs)

file = PATH + data.trn_ds.fnames[1000]
file
img = PIL.Image.open(file) 
img
def get_data(sz, bs):
    tfms = tfms_from_model(arch, sz, aug_tfms = transforms_side_on, max_zoom = 1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', csv_fname = labels_dir,test_name = 'test', suffix = '.jpg',val_idxs = val_idxs,tfms=tfms, bs =bs)
    if sz > 300:
        return data
    else:
        data.resize(340, '/tmp')
        return data
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
data = get_data(sz, bs)
# data = data.resize(int(sz*1.3), '/tmp')
learn = ConvLearner.pretrained(arch, data,tmp_name = TMP_PATH, models_name = MODEL_PATH, precompute = True)

lrf = learn.lr_find()
learn.sched.plot()
%time learn.fit(1e-1,5,cycle_len=1)
learn.precompute = False
%time learn.fit(1e-2,5,cycle_len=1)
learn.save("224_pre")
learn.load("224_pre")
sz = 299 
learn.set_data(get_data(sz,bs))
learn.freeze()
learn.fit(1e-2,3,cycle_len=1)
learn.fit(1e-2, 3, cycle_len = 1, cycle_mult = 2)
from sklearn import metrics
log_preds, y = learn.TTA() 
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y), metrics.log_loss(y, probs)
learn.save("229_pre")
learn.load("229_pre")
log_preds, y = learn.TTA(is_test= True) 
probs = np.mean(np.exp(log_preds), 0)
data.classes 
import pandas as pd
df = pd.DataFrame(probs)
df.columns = data.classes
df.insert(0, 'id', [o[5:-4] for o in data.test_ds.fnames])
df.head()
df.to_csv("Submit.csv",index = False)
        
df.head()
