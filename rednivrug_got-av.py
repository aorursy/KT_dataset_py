import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/got_av/got_av"))
!pip install pretrainedmodels

# !pip install fastai==1.0.50.post1
from torchvision.models import *

import pretrainedmodels



from fastai import *

from fastai.vision import *

from fastai.vision.models import *

from fastai.vision.learner import model_meta

import fastai



from utils import *

import sys

import torch

fastai.__version__
bs = 8
path = "../input/got_av/got_av/train"
filenames = os.listdir('../input/got_av/got_av/test')
df = pd.read_csv('../input/got_av/got_av/train.csv')

df.head()
# CenterCrop(32)

tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0,do_flip=False,xtra_tfms=[cutout()])

data = (ImageList.from_csv(path, csv_name = '../train.csv') 

        .split_by_rand_pct()              

        .label_from_df()            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms, size=256)

        .databunch(num_workers=0,bs=8))
# data
# data.show_batch(rows=3, figsize=(8,10))
# print(data.classes)
# del learn
# learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
# learn.fit_one_cycle(4)
# interp = ClassificationInterpretation.from_learner(learn)



# losses,idxs = interp.top_losses()



# len(data.valid_ds)==len(losses)==len(idxs)
# interp.plot_top_losses(9, figsize=(15,11))
# interp.most_confused(min_val=2)
# learn.save('/kaggle/working/stage-1-34-128')
# learn.fit_one_cycle(6, max_lr=slice(1e-6,1e-4))
# DatasetType.Test
# preds,_ = learn.TTA(ds_type=DatasetType.Test)
def resnext101_32x4d(pretrained=True):

    pretrained = 'imagenet' if pretrained else None

    model = pretrainedmodels.resnext101_64x4d(pretrained=pretrained)

    all_layers = list(model.children())

    return nn.Sequential(*all_layers[0], *all_layers[1:])

learn11 = create_cnn(data, resnext101_32x4d, pretrained=True, metrics=[error_rate, accuracy], model_dir="/tmp/model/",

                   cut=-2, split_on=lambda m: (m[0][6], m[1]))



learn11.fit_one_cycle(6)

learn11.unfreeze()

learn11.lr_find()

learn11.fit_one_cycle(6, max_lr=slice(1e-6,1e-4))
preds2,_ = learn11.TTA(ds_type=DatasetType.Test)
# def noop(*args, **kwargs): return



def get_model(model_name:str, pretrained:bool, seq:bool=False, pname:str='imagenet', **kwargs):

    pretrained = pname if pretrained else None

    model = getattr(pretrainedmodels, model_name)(pretrained=pretrained, **kwargs)

    return nn.Sequential(*model.children()) if seq else model



def resnext101_32x4d(pretrained=True):

    model = get_model('pnasnet5large', pretrained, num_classes=1000)

    #model = pretrainedmodels.pnasnet5large(pretrained='imagenet', num_classes=1000)

    model.logits = noop

    #all_layers = list(model.children())

    return nn.Sequential(model)

learn11 = create_cnn(data, resnext101_32x4d, pretrained=True, metrics=[error_rate, accuracy], model_dir="/tmp/model/",

                    cut=noop,split_on=lambda m: (m[0][0], m[1]))



learn11.fit_one_cycle(6)

learn11.unfreeze()

learn11.lr_find()

learn11.fit_one_cycle(6, max_lr=slice(1e-6,1e-4))
preds3,_ = learn11.TTA(ds_type=DatasetType.Test)
def inceptionresnetv2(pretrained=True):

    pretrained = 'imagenet' if pretrained else None

    model11 = pretrainedmodels.inceptionv4(pretrained=pretrained)

    return nn.Sequential(*model11.children())



learn12 = create_cnn(data, inceptionresnetv2, pretrained=True, metrics=[error_rate, accuracy], model_dir="/tmp/model/",

                   cut=-2, split_on=lambda m: (m[0][11], m[1]))

learn12.fit_one_cycle(6) 

learn12.unfreeze()

learn12.lr_find()

learn12.fit_one_cycle(6, max_lr=slice(1e-6,1e-4))
preds4,_ = learn12.TTA(ds_type=DatasetType.Test)
def inceptionresnetv2(pretrained=True):

    pretrained = 'imagenet' if pretrained else None

    model11 = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)

    return nn.Sequential(*model11.children())



learn12 = create_cnn(data, inceptionresnetv2, pretrained=True, metrics=[error_rate, accuracy], model_dir="/tmp/model/",

                   cut=-2, split_on=lambda m: (m[0][3], m[1]))

learn12.fit_one_cycle(6) 

learn12.unfreeze()

learn12.lr_find()

learn12.fit_one_cycle(6, max_lr=slice(1e-6,1e-4))
preds5,_ = learn12.TTA(ds_type=DatasetType.Test)
torch.save(preds2,'resnext101_64_256')

torch.save(preds3,'pnasnet_256')

torch.save(preds4,'inception_256')

torch.save(preds5,'senet154_300')
# preds,_ = learn.TTA(ds_type=DatasetType.Test)



# preds,_ = learn.get_preds(ds_type=DatasetType.Test)



# pred1 = preds + preds1 + preds2 + preds3 + preds4 + preds5



pred1 = preds3 + preds2 + preds4 + preds5



labelled_preds = []

for pred in pred1:

    labelled_preds.append(int(np.argmax(pred))+1)



submission = pd.DataFrame(

    {'image': filenames,

     'category': labelled_preds,

    })

submission.to_csv('submission.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "subm.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission)