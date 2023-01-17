# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai.metrics import error_rate



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/aerial-cactus-identification/"))
PATH = Path('../input/aerial-cactus-identification/train/')

data_dir = '../input/aerial-cactus-identification/'

train_df = pd.read_csv(os.path.join(data_dir,'train.csv'))

train_df.head()
add_dir = lambda x: os.path.join('train', x)

train_df['id'] = train_df['id'].apply(add_dir)

print(train_df.shape)

data = (ImageList.from_df(train_df,PATH)

        #Where to find the data? 

        .split_by_rand_pct(0.20,seed=44)

        #How to split in train/valid? -> randomly with the default 20% in valid

        .label_from_df()

        #How to label? -> use the second column of the csv file and split the tags by ' '

        #Data augmentation? -> use tfms with a size of 128

        .transform(get_transforms(flip_vert= True),size=32)

        .databunch(bs=8)

        .normalize(imagenet_stats))                          

        #Finally -> use the defaults for conversion to databunch

data
print(data.classes)
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth

learn = cnn_learner(data, models.resnet34, metrics=[accuracy],model_dir='/kaggle',pretrained=True)

learn.summary()
learn.lr_find()

learn.recorder.plot()
lr = 1e-3

learn.fit_one_cycle(3, slice(lr))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-7,1e-6))
sample_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

sample_df.head()
PATH = Path('../input/aerial-cactus-identification/test')

learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test'))

preds,y = learn.get_preds(DatasetType.Test)
sample_df.has_cactus = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)