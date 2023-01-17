# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# train_csv = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

# test_csv = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

# train_csv.head()

# df=train_csv.copy()

# df['id_code'] = train_csv.id_code.apply(lambda x:x+'.png')
from fastai.vision import *

from fastai import *

from fastai.callbacks import *
PATH = Path('../input/aptos2019-blindness-detection')
df_train = pd.read_csv(PATH/'train.csv')

df_test = pd.read_csv(PATH/'test.csv')
#aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])

data = ImageDataBunch.from_df(df=df_train,

                              path=PATH, folder='train_images', suffix='.png',

                              valid_pct=0.1,

                              ds_tfms=get_transforms(flip_vert=False, max_warp=0),

                              size=224,

                              bs=32,

                              seed=37,

                              num_workers=os.cpu_count()

                             ).normalize(imagenet_stats)
# data.show_batch(rows=3, figsize=(7,6))
# !rm -rf model

# !rm -rf ../model
!mkdir /tmp

!mkdir /tmp/.cache/

!mkdir /tmp/.cache/torch/

!mkdir /tmp/.cache/torch/checkpoints

!cp ../input/resnet34fastai/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
os.mkdir('../model')

os.listdir('..')
learn = cnn_learner(data, models.resnet34, metrics=[accuracy,KappaScore(weights="quadratic")],model_dir = '/kaggle/model')

#learn.callbacks.append(SaveModelCallback(learn,monitor='kappa_score', name='best_kappa'))

from fastai.callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback, SaveModelCallback

ES = EarlyStoppingCallback(learn, monitor='kappa_score',patience = 5)

RLR = ReduceLROnPlateauCallback(learn, monitor='valid_loss',patience = 2)

SAVEML = SaveModelCallback(learn, every='improvement', monitor='kappa_score', name='best_kappa')



learn.callbacks.extend([ES,RLR,SAVEML])
learn.callbacks[-1]
learn.freeze()

learn.lr_find();learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=6.31e-3)
learn.unfreeze()

learn.lr_find(); learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=slice(3.31e-6, 1e-4))
tta_params = {'beta':0.12, 'scale':1.0}

sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn = learn.load('best_kappa')
learn.data.add_test(ImageList.from_df(

    sample_df, PATH,

    folder='test_images',

    suffix='.png'

))
# learn.data.test_ds
preds,y = learn.TTA(ds_type=DatasetType.Test, **tta_params)

# num_batch = len(learn.data.test_dl)

# preds,target= learn.get_preds(DatasetType.Test,n_batch=num_batch)

# len(preds)
sample_df.diagnosis = preds.argmax(1)
sample_df.head()
sample_df.to_csv('submission.csv',index=False)