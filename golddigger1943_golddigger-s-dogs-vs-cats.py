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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input/newdogscats/dogscats/dogscats"))
PATH = "../input/newdogscats/dogscats/dogscats/"
# Any results you write to the current directory are saved as output.
os.listdir(PATH)
torch.cuda.is_available()
torch.backends.cudnn.enabled

os.listdir(f'{PATH}valid')
files = os.listdir(f'{PATH}valid/cats')[:5]
files
img = plt.imread(f'{PATH}valid/cats/{files[0]}')
plt.imshow(img);
arch=resnet34
sz=224
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model"
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
lrf=learn.lr_find()
learn.sched.plot_lr()
learn.sched.plot()
learn.fit(1e-2, 1)
learn.fit(1e-2, 3, cycle_len=1)
learn.sched.plot_lr()
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(1e-2, 1)
learn.precompute=False
learn.fit(1e-2, 3, cycle_len=1)
learn.sched.plot_lr()
learn.save('224_lastlayer')
print(os.listdir("/tmp/"))
print(os.listdir("/tmp/tmp"))
print(os.listdir("/tmp/model"))
learn.load('224_lastlayer')
learn.summary()
learn.unfreeze()
learn.summary()
lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()
learn.save('224_all')
learn.load('224_all')
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy_np(probs, y)
preds = np.argmax(probs, axis=1)
probs = probs[:,1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)