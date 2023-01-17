import fastai

from fastai.vision import *
!conda install pytorch torchvision -c pytorch
# import libraries

import torch

import numpy as np

from torchvision import datasets

import torchvision.transforms as transforms
work_dir = Path('/kaggle/working/')

path = Path('../input')
train = 'train_images/train_images'

test =  path/'leaderboard_test_data/leaderboard_test_data'

holdout = path/'leaderboard_holdout_data/leaderboard_holdout_data'

sample_sub = path/'SampleSubmission.csv'

labels = path/'traininglabels.csv'
df = pd.read_csv(labels)

df_sample = pd.read_csv(sample_sub)
df.head()
df.describe()
df[df['score']<0.75]
(df.has_oilpalm==1).sum()
test_names = [f for f in test.iterdir()]

holdout_names = [f for f in holdout.iterdir()]
src = (ImageItemList.from_df(df, path, folder=train)

      .random_split_by_pct(0.2, seed=2019)

      .label_from_df('has_oilpalm')

      .add_test(test_names+holdout_names))
data =  (src.transform(get_transforms(), size=128)

         .databunch(bs=64)

         .normalize(imagenet_stats))
data.show_batch(3, figsize=(10,7))