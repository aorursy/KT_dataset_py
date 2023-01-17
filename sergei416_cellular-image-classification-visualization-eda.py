#imports



import pandas as pd

import json

import os

import sys

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import accuracy_score

import numpy as np

from sklearn.model_selection import train_test_split





!git clone https://github.com/recursionpharma/rxrx1-utils

print ('rxrx1-utils cloned!')

!ls


sys.path.append('rxrx1-utils')

    

from rxrx.main import main

import rxrx.io as rio
t = rio.load_site('train', 'RPE-05', 3, 'D19', 1)



t.shape
fig, axes = plt.subplots(2, 3, figsize=(16, 16))



for i, ax in enumerate(axes.flatten()):

    ax.axis('off')

    ax.set_title('channel {}'.format(i + 1))

    _ = ax.imshow(t[:, :, i], cmap='gray')



    
x = rio.convert_tensor_to_rgb(t)



x.shape
plt.figure(figsize=(8, 8))

plt.axis('off')



_ = plt.imshow(x)




md = rio.combine_metadata()





md.head(10)



md.shape
md.index
md.info()
plt.figure(figsize= (5,10))

plt.subplot(311)

plt.title('cell_type')

plt.tight_layout()

sns.countplot(y =md['cell_type'])
plt.subplot(312)

plt.title('dataset')

plt.tight_layout()

sns.countplot(y = md['dataset'])
plt.subplot(312)

plt.title('plate')

plt.tight_layout()

sns.countplot(y = md['plate'])
plt.subplot(312)

plt.title("site")

plt.tight_layout()

sns.countplot(y = md['site'])
plt.subplot(312)

plt.title('well_type')

plt.tight_layout()

sns.countplot(y = md['well_type'])
#unique values

for i in md.columns:

    print (">> ",i,"\t", md[i].unique())
#Missing values

missing_count = md.isnull().sum()

missing_count
#fill in missing values

md = md.fillna(0)

md.head()
#split into train and test

train_df = md[md["dataset"] == "train"]

test_df = md[md["dataset"] == "test"]

train_df.shape, test_df.shape
#siRNA distribution for train and test sets

plt.figure(figsize=(16,6))

plt.title("Distribution of siRNA in the train and test set")

sns.distplot(train_df.sirna,color="green", kde=True,bins='auto', label='train')

sns.distplot(test_df.sirna,color="blue", kde=True, bins='auto', label='test')

plt.legend()

plt.show()
feat1 = 'sirna'

fig = plt.subplots(figsize=(15, 5))



# train data

plt.subplot(1, 2, 1)

sns.kdeplot(train_df[feat1][train_df['site'] == 1], shade=False, color="b", label = 'site 1')

sns.kdeplot(train_df[feat1][train_df['site'] == 2], shade=False, color="r", label = 'site 2')

plt.title(feat1)

plt.xlabel('Feature Values')

plt.ylabel('Probability')



# test data

plt.subplot(1, 2, 2)

sns.kdeplot(test_df[feat1][test_df['site'] == 1], shade=False, color="b", label = 'site 1')

sns.kdeplot(test_df[feat1][test_df['site'] == 2], shade=False, color="r", label = 'site 2')

plt.title(feat1)

plt.xlabel('Feature Values')

plt.ylabel('Probability')

plt.show()


train_df['category'] = train_df['experiment'].apply(lambda x: x.split('-')[0])

test_df['category'] = test_df['experiment'].apply(lambda x: x.split('-')[0])



train_target_df = pd.get_dummies(train_df['sirna'])







train_df.shape, test_df.shape, train_target_df.shape

train_df.to_csv("train_df.csv")

test_df.to_csv("test_df.csv")

print("done")
# Pixel stats

df_pix = pd.read_csv("../input/pixel_stats.csv")

df_pix.head()
#Flatten the pixels

df_pix['idx'] = df_pix.groupby('id_code').cumcount()

df_pix = df_pix.pivot(index='id_code',columns='idx')[['mean','std', 'median','min','max' ]]

df_pix.columns = df_pix.columns.get_level_values(0)

df_pix.head()
#Only train on treatment well_type

df_pix=df_pix.reset_index()

md=md[md.well_type=='treatment']

md=md.reset_index()



md.head()
df=md[['id_code','sirna', 'dataset','well_type']].merge(df_pix, on='id_code', how='left')

df.head()

df_training = df.loc[df.dataset=='train']

df_test = df.loc[df.dataset=='test']

df_training.shape, df_test.shape


df_train=df_training.drop(['dataset', 'well_type'], axis=1)

df_test=df_test.drop(['dataset','well_type'], axis=1)

#df_train=df_train.drop(['index'], axis=1)

#df_test=df_test.drop(['index'], axis=1)

#df_train.head(), df_test.head()
#create validation set

train, test = train_test_split(df_train, test_size=0.1)

train.shape, test.shape
cols=[]

for i in range(len(train.columns[2:])):

    cols.append(train.columns[i+2]+str(i))

train.columns=['id_code','sirna']+cols

test.columns=['id_code','sirna']+cols

train.head(100)

X = train[train.columns[2:]].copy()

y = train.sirna.values.astype(int)

X_test = test[test.columns[2:]].copy()
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 7).fit(X, y) 
pred_train = knn.predict(X)
accuracy_score(y, pred_train)
test.head()
X.shape, test.shape, test[test.columns[2:]].shape
test['pred']=knn.predict(test[test.columns[2:]])
X.shape, y
df_sub=pd.read_csv("../input/sample_submission.csv")

df_sub.head()



df_submission=df_sub.drop(['sirna'], axis=1).merge(test[['id_code','pred']], on='id_code', how='left')

df_submission.columns=['id_code','sirna']

df_submission.head()
df_submission.to_csv('test_submission.csv',index=False)