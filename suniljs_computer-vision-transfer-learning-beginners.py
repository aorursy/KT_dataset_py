# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

import torch

warnings.filterwarnings("ignore")
#reading locaiton of train images

train=pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

#adding path in the name column

train['name']= train.image_name.apply(lambda x: os.path.join("train",str(x+'.jpg')))

#selectingo only path and label

label_mapping=train[['name','target']]

#similar thing for test data

test=pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

test['name']=test.image_name.apply(lambda x:(str(x+'.jpg')))

test_map=pd.DataFrame(test['name'])
#randomly choosing 1200 images with all events in it. The reason to undersample is to quickly get restults rather than waiting 3hrs on modelling process  

label_mapping_event=label_mapping[label_mapping.target==1]



label_mapping_nonevent=label_mapping[label_mapping.target==0]





label_mapping_ne_sub = label_mapping_nonevent.sample(frac=0.02).reset_index(drop=True)



target_map=pd.concat((label_mapping_event,label_mapping_ne_sub),axis=0).sample(frac=1).reset_index(drop=True)

#importing necessary variables from fastai

%reload_ext autoreload

%autoreload 2

from fastai import *

from fastai.vision import *

np.random.seed(123)

tfms=get_transforms()

test_dl=ImageList.from_df(test_map,path='/kaggle/input/siim-isic-melanoma-classification/jpeg/test')

src=ImageList.from_df(target_map,path="/kaggle/input/siim-isic-melanoma-classification/jpeg").split_by_rand_pct()



data=src.label_from_df().add_test(test_dl).transform(tfms,size=64).databunch(bs=32,num_workers=16).normalize(imagenet_stats)



arch=models.resnet18

learn=cnn_learner(data,arch,metrics=accuracy,model_dir="/kaggle/working/")
lr=0.1/3

learn.fit_one_cycle(1,slice(lr))

learn.save('stage2_model')
learn.load('../input/modelfiles/stage1_model')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5,slice(1e-2))

learn.save('stage_2_model')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3,slice(1e-5))

learn.save('stage_3_model')
from sklearn.metrics import roc_auc_score



def get_auc_subset(preds,targs):

    pred_np=preds[:,1].detach().numpy()

    targ_np=targs.detach().numpy()

    return roc_auc_score(targ_np,pred_np)



def get_roc(learn):

    #pred_train,targ_train=learn.get_preds(ds_type=DatasetType.Train)

    pred_valid,targ_valid=learn.get_preds(ds_type=DatasetType.Valid)

    #train_auc=get_auc_subset(pred_train,targ_train)

    valid_auc=get_auc_subset(pred_valid,targ_valid)

    return valid_auc



valid_auc=get_roc(learn) 


pred_test,_=learn.get_preds(ds_type=DatasetType.Test)



final_pred=pred_test[:,1].detach().numpy()



submission=pd.DataFrame({'target':final_pred})

submission['image_name']=test.image_name

submission=submission[['image_name','target']]

submission.to_csv("submission2.csv",index=False)