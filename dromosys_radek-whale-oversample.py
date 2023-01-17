%matplotlib inline
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
import pandas as pd
from torch import optim
import re

from utils import *
df = pd.read_csv('../input/train.csv')
df.head()
im_count = df[df.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
df = df.join(im_count, on='Id')
val_fns = set(df.sample(frac=1)[(df.Id != 'new_whale') & (df.sighting_count > 1)].groupby('Id').first().Image)
# pd.to_pickle(val_fns, 'data/val_fns')
#val_fns = pd.read_pickle('data/val_fns')
fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)
df = df[df.Id != 'new_whale']
df.shape
df.sighting_count.max()
df_val = df[df.Image.isin(val_fns)]
df_train = df[~df.Image.isin(val_fns)]
df_train_with_val = df
df_val.shape, df_train.shape, df_train_with_val.shape
%%time

res = None
sample_to = 15

for grp in df_train.groupby('Id'):
    n = grp[1].shape[0]
    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)
    rows = pd.concat((grp[1], additional_rows))
    
    if res is None: res = rows
    else: res = pd.concat((res, rows))
%%time

res_with_val = None
sample_to = 15

for grp in df_train_with_val.groupby('Id'):
    n = grp[1].shape[0]
    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)
    rows = pd.concat((grp[1], additional_rows))
    
    if res_with_val is None: res_with_val = rows
    else: res_with_val = pd.concat((res_with_val, rows))
res.shape, res_with_val.shape
pd.concat((res, df_val))[['Image', 'Id']].to_csv('oversampled_train.csv', index=False)
res_with_val[['Image', 'Id']].to_csv('oversampled_train_and_val.csv', index=False)
df = pd.read_csv('oversampled_train.csv')
data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], '../input/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('../input/test'))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input')
        .normalize(imagenet_stats)
)
data

