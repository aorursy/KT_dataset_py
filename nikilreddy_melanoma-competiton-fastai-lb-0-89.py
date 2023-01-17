from fastai.vision import *
from fastai.metrics import error_rate
from fastai.metrics import fbeta
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pathi = Path("../input/siic-isic-224x224-images")
pathc = Path("../input/melanomacsv")
train = pd.read_csv(pathc/'train.csv')
test = pd.read_csv(pathc/'test.csv')
train.head(4)
train.target.value_counts()
train['image_name'] = 'train/' + train['image_name'].astype(str) +'.png'
test['image_name'] = 'test/' + test['image_name'].astype(str) +'.png'
train.head()
test.head()
import warnings
warnings.filterwarnings('ignore')
labelcol=['image_name','target']
labels=train[labelcol]
labels.head(4)
pathi
tfms = get_transforms( flip_vert=True, max_rotate=15, max_zoom=1.2, max_lighting=0.3, max_warp=0, p_affine=0, p_lighting=0.8)
np.random.seed(42)
data = ImageDataBunch.from_df(pathi, labels, ds_tfms=tfms , size=224, bs=48 )
data.normalize(imagenet_stats)
test_data = ImageList.from_df(test,pathi)
data.add_test(test_data)
data
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
len(data.classes)
from fastai.metrics import Precision , accuracy , Recall
from  fastai.metrics import AUROC
pr=Precision()
re=Recall()
aur=AUROC()
!pip install efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b6')
model._fc = nn.Linear(1280,2)
#model = pretrainedmodels.__dict__['resnext101_32x4d'](pretrained=None)
learn = Learner(data, model , metrics=aur ) 
learn.model.cuda();
import torch
torch.cuda.device(0)
torch.cuda.get_device_name(0)
from fastai.utils.mem import GPUMemTrace
with GPUMemTrace():
    learn.fit_one_cycle(12)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
from fastai.utils.mem import GPUMemTrace
with GPUMemTrace():
    learn.fit_one_cycle(4,max_lr=slice(3e-5,3e-3))
'''learn.freeze()
from fastai.utils.mem import GPUMemTrace
with GPUMemTrace():
    learn.fit_one_cycle(4,max_lr=slice(3e-5,3e-4))'''
predictions, *_ = learn.get_preds(DatasetType.Test)
#labels = np.argmax(predictions, 1)
ans =predictions[:,1]
ans
sub=pd.read_csv("../input/submission/sample_submission.csv")
sub
pred=pd.DataFrame(ans)
sub['target']=pred
sub.head()
sub.to_csv('densenet201-8-4.csv',index=False)
