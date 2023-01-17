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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pretrainedmodels
#!git clone https://github.com/Cadene/pretrained-models.pytorch.git
#!cd pretrained-models.pytorch
#!python setup.py install
!pip install utils
!pip install pytorchcv
from torchvision.models import *
from pytorchcv.model_provider import get_model as ptcv_get_model
import pretrainedmodels
import torch
from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta

from utils import *
import sys
[k for k,v in sys.modules['torchvision.models'].__dict__.items() if callable(v)]
pretrainedmodels.model_names
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
df = pd.read_csv('/kaggle/input/dataset/train.csv', header = "infer")
df
df_test = pd.read_csv('/kaggle/input/dataset/test_vc2kHdQ.csv', header = "infer")
df_test
from pathlib import Path
#path = Path('/kaggle/input/dataset/')
#path2 = Path('/kaggle/input/dataset/train_SOaYf6m/images/')
#size = 300
#tfms = get_transforms(do_flip = True, max_lighting = 0.2, max_zoom = 1.1,max_rotate = 45, max_warp = 0.1)
#data = ImageDataBunch.from_csv(path, folder= 'train_SOaYf6m/images', valid_pct = 0.2, csv_labels = 'train_SOaYf6m/train.csv', ds_tfms = tfms, fn_col = 'image_names', label_col = 'emergency_or_not', bs = 16, size = size).normalize(imagenet_stats)
#test_data = ImageList.from_df(df_test, path2)
#data.add_test(test_data)
#data
path = Path('/kaggle/input/dataset/train_SOaYf6m/images/')
size = 300
tfms = get_transforms(do_flip = True, max_lighting = 0.2,max_zoom = 1.1,max_rotate = 45, max_warp = 0.1)
data = ImageDataBunch.from_df(path, df, size=size, bs = 16, ds_tfms = tfms, valid_pct = 0.0).normalize(imagenet_stats)
test_data = ImageList.from_df(df_test, path)
data.add_test(test_data)
#data.normalize(mean = torch.Tensor([0.485, 0.456, 0.406]), std = torch.Tensor([0.229, 0.224, 0.225]))
#data.normalize(imagenet_stats)
data
data.show_batch(rows=3, figsize=(7,6))
def inceptionv4(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.inceptionv4(pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers[0], *all_layers[1:])
def xception(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.xception(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))
def inceptionresnetv2(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.inceptionresnetv2(pretrained=pretrained)
    return nn.Sequential(*model.children())
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
def resnext101_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.resnext101_32x4d(pretrained=pretrained)
    all_layers = list(model.children())
    return nn.Sequential(*all_layers[0], *all_layers[1:])
def efficientnet_b3(pretrained=False):
    return ptcv_get_model("efficientnet_b3", pretrained=False).features
seed = 42

# python RNG
import random
random.seed(seed)

# pytorch RNGs
import torch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np
np.random.seed(seed)
learn = cnn_learner(data, models.resnet101, metrics=[FBeta(beta=1, average='weighted'), accuracy])
#gcontext = ssl.SSLContext()
#learn = cnn_learner(data, resnext101_32x4d, pretrained=True, cut=-2, split_on=lambda m: (m[0][6], m[1]), metrics = accuracy)
#learn = cnn_learner(data, efficientnet_b3, pretrained=True, cut=noop, split_on=lambda m: (m[0][4], m[1]), metrics = accuracy)
#learn = cnn_learner(data, inceptionv4, pretrained=True, cut=-2, split_on=lambda m: (m[0][11], m[1]), metrics = accuracy)
#learn = cnn_learner(data, inceptionresnetv2, pretrained=True, cut=-2, split_on=lambda m: (m[0][9], m[1]), metrics = accuracy)
learn.model_dir = Path('/kaggle/working')
learn.lr_find()
learn.recorder.plot(suggestion = True)
learn.fit_one_cycle(35, max_lr = 3e-4)
#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_top_losses(6)
#interp.plot_confusion_matrix()
#learn2 = cnn_learner(data, inceptionresnetv2, pretrained=True, cut=-2, split_on=lambda m: (m[0][9], m[1]), metrics = accuracy)
#learn2 = cnn_learner(data, resnext101_32x4d, pretrained=True, cut=-2, split_on=lambda m: (m[0][6], m[1]), metrics = accuracy)
learn2 = cnn_learner(data, models.vgg19_bn, metrics=[FBeta(beta=1, average='weighted'), accuracy])
learn2.fit_one_cycle(35, 2e-4)
learn3 = cnn_learner(data, models.resnet50, metrics=[FBeta(beta=1, average='weighted'), accuracy])
learn3.model_dir = Path('/kaggle/working')
learn3.fit_one_cycle(35, 2e-4)
learn4 = cnn_learner(data, models.densenet121, metrics=[FBeta(beta=1, average='weighted'), accuracy])
learn4.model_dir = Path('/kaggle/working')
learn4.fit_one_cycle(35, 2e-4)
learn5 = cnn_learner(data, models.resnet152, metrics=[FBeta(beta=1, average='weighted'), accuracy])
learn5.model_dir = Path('/kaggle/working')
learn5.fit_one_cycle(35, 2e-4)
preds1, _ = learn.get_preds(ds_type = DatasetType.Test)
preds2, _ = learn2.get_preds(ds_type = DatasetType.Test)
preds3, _ = learn3.get_preds(ds_type = DatasetType.Test)
preds4, _ = learn4.get_preds(ds_type = DatasetType.Test)
preds5, _ = learn5.get_preds(ds_type = DatasetType.Test)
pred_ensemble = (preds1 + preds2 + preds3 + preds4 + preds5)/5
pred_ensemble
prob = np.array(pred_ensemble)
preds_fin = np.argmax(prob, axis = 1)
preds_fin
df_test
submission = pd.DataFrame({'image_names' : df_test['image_names'], 'emergency_or_not' : preds_fin})
submission
#Final Solution
submission.to_csv("/kaggle/working/submission_ensemblef.csv", index = False)
#submission = pd.DataFrame({'image_names' : df_test['image_names'], 'emergency_or_not' : np.argmax(preds5, axis = 1)})
#submission.to_csv("/kaggle/working/submission_ind_best.csv", index = False)
#submission