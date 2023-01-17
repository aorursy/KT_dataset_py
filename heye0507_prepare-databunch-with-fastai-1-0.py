import numpy as np

import pandas as pd

import os

from fastai.vision import *

from fastai.basic_data import *
df_label = pd.read_csv('../input/train.csv')

df_label.describe()
src = (ImageItemList.from_csv('../input/','train.csv',folder='train')

        .random_split_by_pct()

        .label_from_df())
src = (ImageItemList.from_csv('../input/','train.csv',folder='train')

        .no_split()

        .label_from_df())
data = (src.transform(get_transforms(),size=224)

       .databunch()

       .normalize(imagenet_stats))
data.show_batch(rows=3,figsize=(12,7))
data = (src.transform(get_transforms(),size=224)

       .databunch(num_workers=0)

       .normalize(imagenet_stats))
data.show_batch(rows=3,figsize=(12,9))
df_test_split = df_label.copy()

df_test_split['total'] = df_test_split.groupby('Id')['Id'].transform('count')

df_grouped = df_test_split.groupby('Id').apply(lambda x: x.sample(frac=0.2,random_state=47))

df_grouped.describe()
df_grouped.tail(10)
df_merged = pd.merge(left=df_test_split,right=df_grouped,on='Image',how='left',suffixes=('','_y'))

df_merged['is_valid'] = df_merged.Id_y.isnull()!=True

df_merged.head(20)
df_merged.drop(['Id_y','total_y'],axis=1,inplace=True)

df_merged.head(10)
df_merged.to_csv('validation_random.csv',index=False)
src = (ImageItemList.from_csv('../input/','/kaggle/working/validation_random.csv',folder='train')

        .split_from_df(col='is_valid')

        .label_from_df(cols='Id'))
data = (src.transform(get_transforms(max_zoom=1, max_warp=0),resize_method=ResizeMethod.SQUISH,size=224)

       .databunch(num_workers=0)

       .normalize(imagenet_stats))
data.show_batch(rows=3,figsize=(12,9))