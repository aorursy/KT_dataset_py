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
import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from tqdm import tqdm_notebook

import warnings

import multiprocessing

warnings.filterwarnings('ignore')

import xgboost as xgb
def load_data(data):

    return pd.read_csv(data)



train = load_data('../input/train.csv')

test = load_data('../input/test.csv')
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

oof = np.zeros(len(train))

preds = np.zeros(len(test))



for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=1.5).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = QuadraticDiscriminantAnalysis(0.7)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')


test['target'] = preds

test.loc[test['target'] > 0.99999, 'target'] = 1

test.loc[test['target'] < 0.00001, 'target'] = 0
usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

new_train = pd.concat([train, usefull_test]).reset_index(drop=True)
oof = np.zeros(len(new_train))

preds = np.zeros(len(test))



for i in tqdm_notebook(range(512)):



    train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=1.5).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=11, random_state=42)

    

    for train_index, test_index in skf.split(train2, train2['target']):

        

        clf = QuadraticDiscriminantAnalysis(0.7)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] = clf.predict_proba(test3)[:,1] / skf.n_splits

auc = roc_auc_score(new_train['target'], oof)

print(f'AUC: {auc:.5}')



sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)