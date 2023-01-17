import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
train2=train[train['wheezy-copper-turtle-magic']==0]



feats = [f for f in train2.columns if f not in ['id','target']]

def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(64,4,figsize=(15,100))



    for feature in features:

        i += 1

        plt.subplot(32,8,i)

        sns.distplot(df1[feature], hist=False,label=label1)

        sns.distplot(df2[feature], hist=False,label=label2)

        

        plt.xlabel(feature, fontsize=9)

        plt.xlim(-30,30)

        plt.ylim(0,1)

        

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();

    

t0 = train2[feats].loc[train['target'] == 0]

t1 = train2[feats].loc[train['target'] == 1]

features = train2[feats].columns.values

plot_feature_distribution(t0, t1, '0', '1', features);
train2=train[train['wheezy-copper-turtle-magic']==0]

min_max = []

for x in train2.columns[1:-1][train2.columns[1:-1]!='wheezy-copper-turtle-magic']:

    min_max.append(train2[x].values.max()-train2[x].values.min())        

sns.distplot(min_max);

plt.title('range histgram (wheezy-copper-turtle-magic=0)')

plt.show()
var = []

for x in train2.columns[1:-1][train2.columns[1:-1]!='wheezy-copper-turtle-magic']:

    var.append(train2[x].var())        

sns.distplot(var);

plt.title('variance histgram (wheezy-copper-turtle-magic=0)')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
oof = np.zeros(len(train))

preds = np.zeros(len(test))

n_split = 5



for i in range(512):

    cols = [c for c in train.columns if c not in ['id', 'target']]

    cols.remove('wheezy-copper-turtle-magic')

    train2 = train[train['wheezy-copper-turtle-magic']==i].copy()

    test2 = test[test['wheezy-copper-turtle-magic']==i].copy()

    

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    test2.reset_index(drop=True,inplace=True)



    skf = StratifiedKFold(n_splits=n_split, random_state=42)

    for train_index, test_index in skf.split(train2[train2.columns[train2.columns!='target']], train2['target']):



        clf = LogisticRegression(solver='sag',penalty='l2',C=0.001)

        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]

        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / n_split



auc = roc_auc_score(train['target'],oof)

print('LR scores CV =',round(auc,5))
oof = np.zeros(len(train))

preds = np.zeros(len(test))

n_split = 5

print('drop columnss whose data range is upper 15.')

for i in range(512):

    cols = [c for c in train.columns if c not in ['id', 'target']]

    cols.remove('wheezy-copper-turtle-magic')

    train2 = train[train['wheezy-copper-turtle-magic']==i].copy()

    test2 = test[test['wheezy-copper-turtle-magic']==i].copy()

    

    ##  Reduction by Range

    for x in cols:

        if train2[x].values.max()-train2[x].values.min() >= 15:

            train2 = train2.drop([x],axis=1)

            test2 = test2.drop([x],axis=1)

            cols.remove(x)

    

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    test2.reset_index(drop=True,inplace=True)



    skf = StratifiedKFold(n_splits=n_split, random_state=42)

    for train_index, test_index in skf.split(train2[train2.columns[train2.columns!='target']], train2['target']):



        clf = LogisticRegression(solver='sag',penalty='l2',C=0.001)

        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]

        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / n_split



auc = roc_auc_score(train['target'],oof)

print('LR with dimention reduction by data range scores CV =',round(auc,5))
oof = np.zeros(len(train))

preds = np.zeros(len(test))

n_split = 5

print('drop columns whose data range is under 15.')

for i in range(512):

    cols = [c for c in train.columns if c not in ['id', 'target']]

    cols.remove('wheezy-copper-turtle-magic')

    train2 = train[train['wheezy-copper-turtle-magic']==i].copy()

    test2 = test[test['wheezy-copper-turtle-magic']==i].copy()

    

    ##  Reduction by Range

    for x in cols:

        if train2[x].values.max()-train2[x].values.min() < 15:

            train2 = train2.drop([x],axis=1)

            test2 = test2.drop([x],axis=1)

            cols.remove(x)

    

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    test2.reset_index(drop=True,inplace=True)



    skf = StratifiedKFold(n_splits=n_split, random_state=42)

    for train_index, test_index in skf.split(train2[train2.columns[train2.columns!='target']], train2['target']):



        clf = LogisticRegression(solver='sag',penalty='l2',C=0.001)

        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]

        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / n_split



auc = roc_auc_score(train['target'],oof)

print('LR with dimention reduction by data range scores CV =',round(auc,5))