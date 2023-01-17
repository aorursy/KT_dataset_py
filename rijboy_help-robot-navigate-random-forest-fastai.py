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
import pandas as pd

import matplotlib.pyplot as plt

% matplotlib inline
train=pd.read_csv('../input/X_train.csv')

test=pd.read_csv('../input/X_test.csv')

y=pd.read_csv('../input/y_train.csv')
train.head()
print('No. of series in training data: ',train['series_id'].nunique())

print('No. of series in testing data: ',test['series_id'].nunique())
train.groupby('series_id')['measurement_number'].count()
(len(train)==128*3810)
test.groupby('series_id')['measurement_number'].count()
(len(test)==3816*128)
test.head()
y.head()
print(len(y)); print(y['surface'].unique()); y['surface'].nunique()
y['surface'].value_counts().reset_index().plot(x='index',y='surface',kind='bar')
def plot_series_distribution(series):

    df_train=train[train['series_id']==series]

    df_test=test[test['series_id']==series]

    plt.figure(figsize=(30,15))

    for i,col in enumerate(df_train.columns[3:]):

        plt.subplot(3,4,i+1)

        df_train[col].hist(bins=100,color='blue')

        df_test[col].hist(bins=100,color='red')

        plt.title(col)
plot_series_distribution(1)
def plot_series(series):

    df_train=train[train['series_id']==series]

    df_test=test[test['series_id']==series]

    plt.figure(figsize=(30,15))

    for i,col in enumerate(df_train.columns[3:]):

        plt.subplot(3,4,i+1)

        df_train[col].plot(color='blue')

        #df_test[col].hist(color='red')

        plt.title(col)

       
plot_series(0)
train_df=train[['series_id']].drop_duplicates().reset_index(drop=True)


test_df=test[['series_id']].drop_duplicates().reset_index(drop=True)
import numpy as np

def new_features(df,tf):

    for i,col in enumerate(df.columns[3:]):

        tf[col+'_mean']=df.groupby('series_id')[col].mean()

        tf[col+'_std']=df.groupby('series_id')[col].std()

        tf[col+'_max']=df.groupby('series_id')[col].max()

        tf[col+'_min']=df.groupby('series_id')[col].min()

        tf[col + '_max_to_min'] = tf[col + '_max'] / tf[col + '_min']

        tf[col+'_abs_max']=df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        tf[col+'_abs_min']=df.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))

        tf[col+'_mad']=df.groupby('series_id')[col].mad()

        tf[col+'_kurtosis']=df.groupby('series_id')[col].apply(lambda x: x.kurtosis())

        tf[col+'_skew']=df.groupby('series_id')[col].skew()

        tf[col+'_median']=df.groupby('series_id')[col].median()

        tf[col+'_rolling_avg_10']=df.groupby('series_id')[col].rolling(10).mean().mean(skipna=True)

        tf[col+'_rolling_avg_10']=df.groupby('series_id')[col].rolling(10).std().mean(skipna=True)

    return tf
train_df=new_features(train,train_df)
test_df=new_features(test,test_df)
test_df.head()
new_train=train_df.copy()

new_test=test_df.copy()
!pip install fastai==0.7.0
%load_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.imports import *

from fastai.structured import *

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
new_train['surface']=y['surface']
train_cats(new_train)
df_train,y,nas=proc_df(new_train,'surface')

df_test,_,nas=proc_df(new_test,na_dict=nas)

df_train,y,nas=proc_df(new_train,'surface',na_dict=nas)
def rmse(x,y):

    return np.sqrt(((x-y)**2).mean())

def print_score(m):

    res=[rmse(m.predict(X_train),y_train),rmse(m.predict(X_valid),y_valid),m.score(X_train,y_train),m.score(X_valid,y_valid)]

    if hasattr(m,'oob_score_'):

        res.append(m.oob_score_)

    print(res)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_train, y, test_size = 0.2, random_state = 42)
m=RandomForestClassifier(n_estimators=120,min_samples_leaf=5,max_features=0.5,oob_score=True,n_jobs=-1)

m.fit(X_train,y_train)

pred=m.predict(X_valid)

print_score(m)
m=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features=0.5,oob_score=True,n_jobs=-1)

m.fit(X_train,y_train)

pred=m.predict(X_valid)

#print(accuracy_score(y_valid,pred))

print_score(m)
plt.figure(figsize=(16,16))

fi=rf_feat_importance(m,df_train)

fi.plot(kind='bar',x='cols',y='imp')

df_k=fi[fi['imp']>=0.005]['cols'] # We'll keep features having feature importance value greater than 0.005
len(df_k)
df_keep=df_train[df_k]

df_keep_test=df_test[df_k]
X_train, X_valid, y_train, y_valid = train_test_split(df_keep, y, test_size = 0.2, random_state = 42)

X_train.shape
m=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features=0.5,oob_score=True,n_jobs=-1)

m.fit(X_train,y_train)

pred=m.predict(X_valid)

print_score(m)
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,16))

dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)

plt.show()
def get_oob(df):

    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, oob_score=True)

    X_train, X_valid, y_train, y_valid = train_test_split(df_keep, y, test_size = 0.2, random_state = 42)

    m.fit(X_train, y_train)

    return m.oob_score_
get_oob(df_keep)
a=['orientation_Z_mean','orientation_Z_median','orientation_Z_min','orientation_Z_max','orientation_Y_mean','orientation_Y_median','orientation_Y_min','orientation_Y_max','orientation_X_mean','orientation_X_median','orientation_X_min','orientation_X_max','orientation_W_mean','orientation_W_median','orientation_W_min','orientation_W_max']
for c in a:

    print(c,get_oob(df_keep.drop(c,axis=1)))
to_drop=['orientation_Z_mean','orientation_Z_median','orientation_Y_mean','orientation_Y_median','orientation_X_mean','orientation_X_median','orientation_W_mean','orientation_W_median']
get_oob(df_keep.drop(to_drop, axis=1))
df_keep.drop(to_drop, axis=1, inplace=True)

df_keep_test.drop(to_drop,axis=1,inplace=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)

sub_preds_rf = np.zeros((df_keep_test.shape[0], 9))

oof_preds_rf = np.zeros((df_keep.shape[0]))

score = 0

for i, (train_index, test_index) in enumerate(folds.split(df_keep, y)):

    print('-'*20, i, '-'*20)

    

    clf =  RandomForestClassifier(n_estimators = 200, n_jobs = -1)

    clf.fit(df_keep.iloc[train_index], y[train_index])

    oof_preds_rf[test_index] = clf.predict(df_keep.iloc[test_index])

    sub_preds_rf += clf.predict_proba(df_keep_test) / folds.n_splits

    score += clf.score(df_keep.iloc[test_index], y[test_index])

    print('score ', clf.score(df_keep.iloc[test_index], y[test_index]))

    
sub_preds_rf
sub_preds_rf.argmax(axis=1)
new_train['surface'].cat.codes.unique()
new_train['surface'].cat.categories
s={0:'carpet',1:'concrete',2:'fine_concrete',3:'hard_tiles',4:'hard_tiles_large_space',5:'soft_pvc',6:'soft_tiles',7:'tiled',8:'wood'}

df_sub=pd.DataFrame({'series_id':df_test['series_id'],'predictions':sub_preds_rf.argmax(axis=1)})
df_sub['surface']=df_sub['predictions'].apply(lambda x: s[x])
df_sub.drop(['predictions'],axis=1,inplace=True)
df_sub.to_csv('submission2_cv.csv',index=False)