import sys

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from skimage.io import imread

from sklearn.metrics import accuracy_score

%matplotlib inline
import os

import sys

!git clone https://github.com/recursionpharma/rxrx1-utils



sys.path.append('rxrx1-utils')

import rxrx.io as rio



md = rio.combine_metadata()



md.head()
t = rio.load_site('train', 'RPE-05', 3, 'D19', 2)

fig, axes = plt.subplots(2, 3, figsize=(24, 16))



for i, ax in enumerate(axes.flatten()):

    ax.axis('off')

    ax.set_title('channel {}'.format(i + 1))

    ax.imshow(t[:, :, i], cmap='gray')
y = rio.load_site_as_rgb('train', 'HUVEC-08', 4, 'K09', 1)



plt.figure(figsize=(8, 8))

plt.axis('off')



plt.imshow(y)
md = rio.combine_metadata()

md.head()
df_pix = pd.read_csv(BASE_PATH+'pixel_stats.csv')

df_pix.head()
df_pix['idx'] = df_pix.groupby('id_code').cumcount()

df_pix = df_pix.pivot(index='id_code',columns='idx')[['mean','std', 'median','min','max' ]]

df_pix.columns = df_pix.columns.get_level_values(0)

df_pix.head()
df_pix=df_pix.reset_index()

md=md[md.well_type=='treatment']

md=md.reset_index()



md.head()
df=md[['id_code','sirna', 'dataset','well_type']].merge(df_pix, on='id_code', how='left')
df_train = df.loc[df.dataset=='train']

df_test = df.loc[df.dataset=='test']

df_train.shape, df_test.shape
df_train=df_train.drop(['dataset', 'well_type'], axis=1)

df_test=df_test.drop(['dataset','well_type'], axis=1)
cols=[]

for i in range(len(df_train.columns[2:])):

    cols.append(df_train.columns[i+2]+str(i))

df_train.columns=['id_code','sirna']+cols

df_test.columns=['id_code','sirna']+cols
X = df_train[df_train.columns[2:]].copy()

y=df_train.sirna.values.astype(int)

X_test = df_test[df_test.columns[2:]].copy()
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 7).fit(X, y) 
pred_train = knn.predict(X)
accuracy_score(y, pred_train)
df_test.head()
X.shape, df_test.shape, df_test[df_test.columns[2:]].shape
df_test['pred']=knn.predict(df_test[df_test.columns[2:]])
df_sub=pd.read_csv(BASE_PATH+'sample_submission.csv')

df_sub.head()



df_submission=df_sub.drop(['sirna'], axis=1).merge(df_test[['id_code','pred']], on='id_code', how='left')

df_submission.columns=['id_code','sirna']

df_submission.head()
df_submission.to_csv('test_submission.csv',index=False)