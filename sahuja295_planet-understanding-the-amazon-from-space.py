%reload_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import warnings
from fastai.conv_learner import *
from IPython.core.interactiveshell import InteractiveShell
print("CUDA is available=" + str(torch.cuda.is_available()))
warnings.filterwarnings("ignore", category=DeprecationWarning)
InteractiveShell.ast_node_interactivity = "all"
PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=64;bs=64
f_model = resnet34
print((os.listdir("../input/")))
print(len(os.listdir("../input/test-jpg-v2")))
print(len(os.listdir("../input/train-jpg")))
train_data = pd.read_csv("../input/train_v2.csv")
label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)
def get_data(sz):
    tfms = tfms_from_model(f_model, sz,aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg',label_csv, tfms=tfms, suffix='.jpg',val_idxs=val_idxs, test_name='test-jpg-v2')
data = get_data(256)
img = PIL.Image.open(PATH + data.trn_ds.fnames[0])
plt.imshow(img)
print("train={},valid={},test={}".format(len(data.trn_ds.fnames),len(data.val_ds.fnames),len(data.test_ds.fnames)))
x,y = next(iter(data.val_dl))
x.shape,y.shape
#first image and # its lables
x[0].shape,y[0]
list(zip(data.classes, y[0]))
plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.0)
plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);
sz=64
data = get_data(sz)
#data = data.resize(int(sz*1.3), '/tmp')
from sklearn.metrics import fbeta_score
import warnings

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])
metrics=[f2]
f_model = resnet34
learn = ConvLearner.pretrained(f_model, data, metrics=metrics, tmp_name=TMP_PATH, models_name=MODEL_PATH)
lrf=learn.lr_find()
learn.sched.plot()
lr = 0.2
lrs = np.array([lr/9,lr/3,lr])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_loss()
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
sz=64
#learn.load(f'{sz}')
learn.save(f'{sz}')
learn.sched.plot_loss()
sz=128
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')
learn.sched.plot_loss()
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')
sz=256
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')
multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)
f2(preds,y)