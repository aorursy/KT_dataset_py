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



from fastai.vision import *
TRAIN_DIR = '../input/train_images/'

TEST_DIR = '../input/test_images/'
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.shape,test_df.shape
train_df.head()
train_df.category_id.value_counts()
sample_train_df = pd.concat([train_df.iloc[:1000],train_df.loc[train_df['category_id']==22]])

sample_train_df.shape
sample_train_df.category_id.value_counts()
# %time train = ImageList.from_df(df=train_df,path=TRAIN_DIR,cols='file_name')

# %time test = ImageList.from_df(df=test_df,path=TEST_DIR,cols='file_name')



%time train = ImageList.from_df(df=sample_train_df,path=TRAIN_DIR,cols='file_name')

%time test = ImageList.from_df(df=test_df,path=TEST_DIR,cols='file_name')
%%time

data = (train.split_by_rand_pct(seed=22)

       .label_from_df(cols='category_id')

       .add_test(test)

       .transform(get_transforms(),size=256)

       .databunch())
data.show_batch()
data.batch_size
len(data.train_ds), len(data.valid_ds), len(data.test_ds),
learn = cnn_learner(data, models.resnet50, metrics=[FBeta(),accuracy], model_dir='../working/model/',path='../working/tmp/')
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 6e-3

learn.fit_one_cycle(2, slice(lr))
lr = 6e-3

learn.fit_one_cycle(2, slice(lr))
learn.save('stage_1_sz256_resnet50')