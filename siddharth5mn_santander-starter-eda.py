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

import numpy as np

from os import listdir

from os.path import isfile, join

from IPython.display import display

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import warnings

import random as rnd

from sklearn import preprocessing

from sklearn.cluster import MiniBatchKMeans

from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

import xgboost as xgb

import lightgbm as lgb

from sklearn import metrics

from sklearn.model_selection import train_test_split

import scipy

import matplotlib.pyplot as plt

from scipy.cluster import hierarchy as hc
path ="../input"
allfiles = [f for f in listdir(path) if isfile(join(path, f)) if (not f.endswith('.ipynb'))] 

allfiles
def print_size(file):

    statinfo=os.stat(join(path,file))

    print('Size of file {} in Bytes is {} bytes'.format(str(file),str(statinfo.st_size)))
for f in allfiles:

    print_size(f)
def load_df(df,**kwds):

    if str(df).endswith('.csv'):

        ndf =pd.read_csv(str(df),**kwds)

    else:

        print('Could not find the df')  

    print('Loaded file {} with shape{}'.format(df,ndf.shape))

    return ndf
test =load_df(join(path,allfiles[0]))

train =load_df(join(path,allfiles[1]))

submission =load_df(join(path,allfiles[2]))
train.head()
train.target.value_counts()
train.describe(include='all').T
train_df =train.drop(['ID_code','target'],axis =1)
train_df.isnull().sum().sort_index()/len(train_df)
def auc(x,y): return metrics.roc_auc_score(x,y)



def print_score(m):

    res = [auc(y_train,m.predict_proba(X_train)[:,1]),auc(y_valid,m.predict_proba(X_valid)[:,1]),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    print(res)
df_trn, y_trn =train_df, train['target']

df_tst = test.drop(['ID_code'], axis=1)

X_train, X_valid,y_train, y_valid = train_test_split(df_trn,y_trn, test_size =.20, stratify =y_trn,random_state =2019)

X_train.shape, y_train.shape, X_valid.shape
m = RandomForestClassifier(n_estimators=10, n_jobs=8,min_samples_leaf=3, max_features=0.5)

m.fit(X_train, y_train)

print_score(m)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, df_trn); fi[:30]
fi.plot('cols', 'imp', figsize=(10,6), legend=False)
def plot_fi(fi): 

    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:100])
plot_fi(fi[:30])
to_keep = fi[fi.imp>0.005].cols; len(to_keep)
df_keep = df_trn[to_keep].copy()

X_train, X_valid,y_train, y_valid = train_test_split(df_trn,y_trn, test_size =.20, stratify =y_trn,random_state =2019)
plt.figure(figsize=(26, 24))

for i, col in enumerate(list(df_keep.columns)[:24]):

    plt.subplot(6, 4, i + 1)

    plt.hist(train[col])

    plt.title(col)
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)

plt.show()
class Model(object):

    def __init__(self,trn,tst,tget,subm):

        self.trn = trn

        self.tst = tst

        self.tget = tget

        self.subm =subm



    def predict(self, path, model):

        X_train, X_val, y_train, y_val = self.kfold_cv()

        

        y_pred = model(X_train, y_train, X_val, y_val)

        self.save_pred(y_pred, path)



    def kfold_cv(self):

        X = self.trn.values

        y = self.tget.values



        kf = KFold(n_splits=5, random_state=2019)

        kf.get_n_splits(X)

        for train_index, val_index in kf.split(X):

            X_train, X_val = X[train_index], X[val_index]

            y_train, y_val = y[train_index], y[val_index]



        return X_train, X_val, y_train, y_val

    

    def xgb(self, X_train, y_train, X_val, y_val):



        dtrain = xgb.DMatrix(X_train, label = y_train)

        dvalid = xgb.DMatrix(X_val, label = y_val)

        dtest = xgb.DMatrix(self.tst.values)



        dlist = [(dtrain, 'train'), (dvalid, 'valid')]



        parameters = {'eta': 0.025,

                    'colsample_bytree': 0.7,

                    'max_depth': 4,

                    'subsample': 0.7,

                    'nthread': -1,

                    'booster' : 'gbtree',

                    'colsample_bytree': 0.7,

                    'min_child_weight': 50,  

                    'colsample_bylevel': 0.7,

                    'lambda' :1, 

                    'alpha':0,

                    'eval_metric' : "auc",

                    'silent': 1,

                    'objective': 'binary:logistic'}



        model = xgb.train(parameters, dtrain, 20000, dlist, early_stopping_rounds=300,

                        verbose_eval=100)



        print("Start prediction ...")



        y_pred = model.predict(dtest)



        return y_pred

    

    def lgb(self, X_train, y_train, X_val, y_val):



        dtrain = lgb.Dataset(X_train, label=y_train)

        dvalid = lgb.Dataset(X_val, label=y_val)

        dtest = self.tst.values

        dlist = [dtrain, dvalid]



        parameters = {'learning_rate': 0.025,

                    'colsample_bytree': 0.3,

                    'max_depth': 6,

                    'subsample': 0.9,

                    'num_threads': -1,

                    'boosting_type' : 'gbdt',

                    'metric' : 'auc',

                    'objective': 'binary',

                    'num_leaves' : 20,

                    'sub_feature':0.7,

                    'sub_row' :0.7, 

                    'bagging_freq':1,

                    'lambda_l1':5, 

                    'lambda_l2': 5}



        model = lgb.train(parameters, dtrain, 20000, dlist,early_stopping_rounds=300, verbose_eval=100)



        print("Start prediction ...")

        y_pred = model.predict(dtest)



        return y_pred

    

    def save_pred(self, y_pred, path):

        self.subm['target'] = y_pred

        self.subm[['ID_code','target']].to_csv(path, index=False)
print("Training model...\n")

model = Model(df_trn,df_tst,y_trn,submission)

y_pred = model.predict(path ='submission_lgb.csv', model=model.lgb)
print("Training model...\n")

model = Model(df_trn,df_tst,y_trn,submission)

y_pred = model.predict(path ='submission_xgb.csv', model=model.xgb)