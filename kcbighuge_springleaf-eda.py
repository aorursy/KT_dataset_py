import os
import numpy as np
import pandas as pd 
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import seaborn
def autolabel(arrayA):
    ''' label each colored square with the corresponding data value. 
    If value > 20, the text is in black, else in white.
    '''
    arrayA = np.array(arrayA)
    for i in range(arrayA.shape[0]):
        for j in range(arrayA.shape[1]):
                plt.text(j,i, "%.2f"%arrayA[i,j], ha='center', va='bottom',color='w')

def hist_it(feat):
    '''Plot histogram of features with label'''
    plt.figure(figsize=(16,4))
    feat[Y==0].hist(bins=range(int(feat.min()),int(feat.max()+2)),normed=True,alpha=0.8)
    feat[Y==1].hist(bins=range(int(feat.min()),int(feat.max()+2)),normed=True,alpha=0.5)
    plt.ylim((0,1))
    
def gt_matrix(feats,sz=16):
    '''Plot heatmap of features with values greater than other features'''
    a = []
    for i,c1 in enumerate(feats):
        b = [] 
        for j,c2 in enumerate(feats):
            mask = (~train[c1].isnull()) & (~train[c2].isnull())
            if i>=j:
                b.append((train.loc[mask,c1].values>=train.loc[mask,c2].values).mean())
            else:
                b.append((train.loc[mask,c1].values>train.loc[mask,c2].values).mean())

        a.append(b)

    plt.figure(figsize = (sz,sz))
    plt.imshow(a, interpolation = 'None')
    _ = plt.xticks(range(len(feats)),feats,rotation = 90)
    _ = plt.yticks(range(len(feats)),feats,rotation = 0)
    autolabel(a)
def hist_it(feat):
    '''Plot histogram with 100 bins'''
    plt.figure(figsize=(16,4))
    feat[Y==0].hist(bins=100,range=(feat.min(),feat.max()),normed=True,alpha=0.5)
    feat[Y==1].hist(bins=100,range=(feat.min(),feat.max()),normed=True,alpha=0.5)
    plt.ylim((0,1))
!ls ../input
PATH = '../input/'
train = pd.read_csv(PATH + 'train.csv.zip')

# reduce size of data to prevent kernel crashes
SAMPLE_SIZE = 1000
rand_idx = np.random.randint(0, len(train), size=SAMPLE_SIZE)
train = train.iloc[rand_idx,]
train.shape, train.head()

Y = train.target
Y.head()
# Loading too much data might cause the kerenel to crash
# Convert this to a code cell if you want to risk it
test = pd.read_csv(PATH + 'test.csv.zip')

# reduce size of data to prevent kernel crashes
rand_idx = np.random.randint(0, len(test), size=SAMPLE_SIZE)
test = test.iloc[rand_idx,]
test_ID = test.ID
test_ID.head()
import gc; gc.collect()
train.shape, train.head()
# Loading too much data might cause the kerenel to crash
# Convert this to a code cell if you want to risk it
test.shape, test.head()
# Number of NaNs for each object
train.isnull().sum(axis=1).head(12)
# Number of NaNs for each column
train.isnull().sum(axis=0).head(12)
# combining datasets seems to crash the kernel
traintest = pd.concat([train, test], axis = 0)
traintest.shape
# `dropna = False` makes nunique treat NaNs as a distinct value
feats_counts = train.nunique(dropna = False)
feats_counts.sort_values()[:10]
constant_features = feats_counts.loc[feats_counts==1].index.tolist()
print (constant_features)

# Loading too much data might cause the kerenel to crash
traintest = traintest.drop(constant_features, axis = 1)
traintest.shape
traintest = traintest.fillna('NaN')
traintest.head()
train_enc =  pd.DataFrame(index = train.index)

for col in tqdm_notebook(traintest.columns):
    train_enc[col] = train[col].factorize()[0]

train_enc.shape, train_enc.head()
# train_enc[col] = train[col].map(train[col].value_counts())
dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1
dup_cols.items()
# might have to install cPickel
#!pip install cPickle
#import cPickle as pickle
#pickle.dump(dup_cols, open('dup_cols.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
traintest = traintest.drop(dup_cols.keys(), axis = 1)
traintest.shape, traintest.head()
nunique = train.nunique(dropna=False)
nunique[:10]
plt.figure(figsize=(14,4))
plt.hist(nunique.astype(float)/train.shape[0], bins=80, orientation='horizontal');
mask = (nunique.astype(float)/train.shape[0] > 0.8)
train.loc[:train.index[5], mask]
train_idx_orig = train.index
train = train.reset_index(drop=True)
Y = Y.reset_index(drop=True)
train.head(), Y.head()
mask = (nunique.astype(float)/train.shape[0] < 0.8) & (nunique.astype(float)/train.shape[0] > 0.4)
train.loc[:10, mask]
train['VAR_0015'].value_counts()
cat_cols = list(train.select_dtypes(include=['object']).columns)
num_cols = list(train.select_dtypes(exclude=['object']).columns)
train = train.replace('NaN', -999)
# select first few numeric features
feats = num_cols[:32]

# build 'mean(feat1 > feat2)' plot
gt_matrix(feats,16)
hist_it(train['VAR_0002'])
plt.ylim((0,0.05))
plt.xlim((-10,1010));
hist_it(train['VAR_0003'])
plt.ylim((0,0.03))
plt.xlim((-10,1010));
train['VAR_0002'].value_counts()[:10]
train['VAR_0003'].value_counts()[:10]
train['VAR_0004_mod50'] = train['VAR_0004'] % 50
hist_it(train['VAR_0004_mod50'])
plt.ylim((0,0.6))
train.loc[:,cat_cols].head().T
date_cols = [u'VAR_0073','VAR_0075',
             u'VAR_0156',u'VAR_0157',u'VAR_0158','VAR_0159',
             u'VAR_0166', u'VAR_0167',u'VAR_0168',u'VAR_0169',
             u'VAR_0176',u'VAR_0177',u'VAR_0178',u'VAR_0179',
             u'VAR_0204',
             u'VAR_0217']

for c in date_cols:
    train[c] = pd.to_datetime(train[c],format = '%d%b%y:%H:%M:%S')
    test[c] = pd.to_datetime(test[c],  format = '%d%b%y:%H:%M:%S')
c1 = 'VAR_0217'
c2 = 'VAR_0073'

# mask = (~test[c1].isnull()) & (~test[c2].isnull())
# sc2(test.ix[mask,c1].values,test.ix[mask,c2].values,alpha=0.7,c = 'black')

mask = (~train[c1].isnull()) & (~train[c2].isnull())
plt.figure(figsize=(14,4))
plt.scatter(train.loc[mask,c1].values,train.loc[mask,c2].values, c=train.loc[mask,'target'].values, 
            alpha=.5);