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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
path="../input/aptos2019-blindness-detection/"
!ls ../input/resnet50/
# copy pretrained weights for resnet50 to the folder fastai will search by default

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'
%%time

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

train_df.head()

test_df.head()
train_df.head()
test_df.head()
train_img=open_image('../input/aptos2019-blindness-detection/train_images/002c21358ce6.png')

#train_img
train_img.size
test_img=open_image('../input/aptos2019-blindness-detection/test_images/10407824638c.png')

#test_img
test_img.size
train_df['diagnosis'].hist(figsize = (10, 5))
len(train_df.index)
#!ls ../input/aptos2019-blindness-detection/train_images
#tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.2,max_lighting=0.1,p_lighting=0.5) 
bs = 64 #smaller batch size is better for training, but may take longer

sz=224
np.random.seed(42)

src = (ImageList.from_csv(path, 'train.csv', folder='train_images', suffix='.png')

       .split_by_rand_pct(0.2)

       .label_from_df())
data= (src.transform(tfms, size=sz, padding_mode='zeros') #Data augmentation

            .databunch(bs=bs, num_workers=2) #DataBunch

            .normalize(imagenet_stats) #Normalize     

           )
print(data.classes);data.c
data.show_batch(rows=6, figsize=(12,9))
arch = models.resnet50
MODEL_PATH = "../model/"
learn = cnn_learner(data, arch, metrics=[error_rate,accuracy,KappaScore(weights="quadratic")],model_dir=MODEL_PATH)
#learn.freeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr=3e-3
data.train_ds[0][0].shape
learn.fit_one_cycle(10,max_lr = lr)

#learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50')
learn.load('stage-1-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8, max_lr = slice(2e-5,lr/5))
learn.recorder.plot_losses()
learn.save('stage-2-rn50')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix()
learn.recorder.plot_lr(show_moms=True)

sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(

    sample_df, '../input/aptos2019-blindness-detection/',

    folder='test_images',

    suffix='.png'

))
preds,y = learn.TTA(ds_type=DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)

_ = sample_df.hist()