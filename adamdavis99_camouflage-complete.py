# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
from fastai.vision.learner import *
from fastai.vision.models import *


path=Path('../input/normal-vs-camouflage-clothes/8k_normal_vs_camouflage_clothes_images')
path.ls()
bs=64
np.random.seed(42)
data=ImageDataBunch.from_folder(path,valid_pct=0.2,ds_tfms=get_transforms(),size=bs).normalize(imagenet_stats)
      
data.show_batch(3,figsize=(7,6))
data.show_batch(4,figsize=(7,6))
print(data.classes)
print(data.c)
print(len(data.train_ds))
print(len(data.valid_ds))
path=Path('/kaggle/input/pretrained-pytorch-models/resnet50-19c8e357.pth')
path.cwd()
!cp /kaggle/input/pretrained-pytorch-models/resnet50-19c8e357.pth /kaggle/working
!cp /kaggle/working/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth
arch=resnet50
learn = cnn_learner(data,arch,  metrics=[error_rate,accuracy],model_dir='/kaggle/working').to_fp16()
learn.model_dir="/kaggle/working"
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr
lr
torch.cuda.is_available()

torch.backends.cudnn.enabled
learn.fit_one_cycle(5,lr)
learn.save('stage-1-50')
learn.load('stage-1-50')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr1 = learn.recorder.min_grad_lr
lr1
learn.fit_one_cycle(5, max_lr=slice(lr1/100,lr1/10,lr1))
learn.save('model-2')
learn.load('model-2')
interp=ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(4)
interp.plot_confusion_matrix()
def accuracy_topk(output, target, topk=(3,)):
    maxk = max(topk)
    batch_size = target.size(0)
   
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
 
    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
   
   
    
output,target=learn.get_preds()
print(output)
learn.validate(learn.data.valid_dl)
learn.show_results()
learn.export('/kaggle/working/trained_model.pkl')
