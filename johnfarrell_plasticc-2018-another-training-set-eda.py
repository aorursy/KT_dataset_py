%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
pb = [0, 1, 2, 3, 4, 5]
# choice = 'extragalactic'
# choice = 'galactic'
choice = 'all'
train = pd.read_csv('../input/training_set.csv')
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train_meta.columns
extra_cols = ['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod']
gal_mask = train_meta['distmod'].isnull().values #galactic
print(train_meta.shape, gal_mask.sum())
print(train_meta['target'].unique())

print(f'Select {choice}')
if choice=='galactic':
    train_meta = train_meta[gal_mask]
    train_meta.drop(extra_cols, axis=1, inplace=True)
elif choice=='extragalactic':
    train_meta = train_meta[~gal_mask]
else:
    pass

print(train_meta.shape, gal_mask.sum())
print(train_meta['target'].unique())
df = train.merge(train_meta, on='object_id', how='inner').reset_index(drop=True)
target = train_meta[['object_id', 'target']].copy()
print(df.shape)
display(df.head(10))
grps = df.groupby('object_id')
tmp = target.merge(grps.size().rename('obj_size').reset_index(), on='object_id')
sns.barplot(x='target', y='obj_size', data=tmp)
def plot_agg(col, func, grps=grps, target=target):
    tmp = grps.agg({col:func})
    tmp.columns = [f'{col}_{func}']
    tmp = target.merge(tmp, on='object_id')
    sns.boxplot(x='target', y=f'{col}_{func}', data=tmp)
    plt.grid()
    if tmp[f'{col}_{func}'].max()>1000:
        plt.yscale('log')
func_li = ['mean', 'std']
cols = df.columns.tolist()
cols.remove('object_id')
cols.remove('target')
print(cols)
from itertools import product
pairs = list(product(cols, func_li))
plt.figure(figsize=[24, 28])
cnt = 0
for i,(col,func) in enumerate(pairs):
    if col in train_meta and func=='std':
        continue
    else:
        plt.subplot(7, 3, cnt+1)
        plot_agg(col, func)
        cnt += 1
plt.figure(figsize=[18, 4])
sns.distplot(df['mjd'], bins=100)
plt.grid();
plt.figure(figsize=[16, 5])
plt.subplot(1,2,1)
tmp = grps['mjd'].apply(lambda x: x.max()-x.min())
tmp = tmp.rename('mjd_length').reset_index()
tmp = target.merge(tmp, on='object_id')
sns.boxplot(x='target', y=f'mjd_length', data=tmp)
plt.grid()

plt.subplot(1,2,2)
tmp = grps['mjd'].apply(lambda x: (x%1).mean())
tmp = tmp.rename('day_mean').reset_index()
tmp = target.merge(tmp, on='object_id')
sns.boxplot(x='target', y=f'day_mean', data=tmp)
plt.grid()
tmp = df['mjd']%1
tmp.to_frame().describe().T
tmp[tmp<=0.5].min(), tmp[tmp<=0.5].max(), tmp[tmp>0.5].min(), tmp[tmp>0.5].max()
(86400*tmp[tmp<0.5].max())/3600, (86400*tmp[tmp>0.5].min())/3600
plt.figure(figsize=[18, 3])
sns.distplot(df['mjd']%1, bins=100)
plt.grid();
plt.title('Hide during the day, come out at night')
plt.xlabel('Time within a day');