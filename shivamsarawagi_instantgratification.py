# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
cols = [c for c in train_df.columns if c not in ['id', 'target']]

print(cols)
#cols.remove('wheezy-copper-turtle-magic')

interactions = np.zeros((512,255))

oof = np.zeros(len(train_df))

preds = np.zeros(len(test_df))
for i in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train_df[train_df['wheezy-copper-turtle-magic']==i]

    test2 = test_df[test_df['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    test2.reset_index(drop=True,inplace=True)

    

    skf = StratifiedKFold(n_splits=25, random_state=42)

    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):

        # LOGISTIC REGRESSION MODEL

        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)

        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])

        oof[idx1[test_index]] += clf.predict_proba(train2.loc[test_index][cols])[:,1]

        preds[idx2] += clf.predict_proba(test2[cols])[:,1]

        # RECORD INTERACTIONS

        for j in range(255):

            if clf.coef_[0][j]>0: interactions[i,j] = 1

            elif clf.coef_[0][j]<0: interactions[i,j] = -1

    if i%25==0: print(i)
# PRINT CV AUC

auc = roc_auc_score(train_df['target'],oof)

print('LR with interactions scores CV =',round(auc,5))
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



plt.figure(figsize=(15,5))



# PLOT ALL ZIPPY

plt.subplot(1,2,1)

sns.distplot(train_df[ (train_df['target']==0) ]['zippy-harlequin-otter-grandmaster'], label = 't=0')

sns.distplot(train_df[ (train_df['target']==1) ]['zippy-harlequin-otter-grandmaster'], label = 't=1')

plt.title("Without interaction, zippy has no correlation \n (showing all rows)")

plt.xlim((-5,5))

plt.legend()



# PLOT ZIPPY WHERE WHEEZY-MAGIC=0

plt.subplot(1,2,2)

sns.distplot(train_df[ (train_df['wheezy-copper-turtle-magic']==0) & (train_df['target']==0) ]

             ['zippy-harlequin-otter-grandmaster'], label = 't=0')

sns.distplot(train_df[ (train_df['wheezy-copper-turtle-magic']==0) & (train_df['target']==1) ]

             ['zippy-harlequin-otter-grandmaster'], label = 't=1')

plt.title("With interaction, zippy has postive correlation \n (only showing rows where wheezy-copper-turtle-magic=0)")

plt.legend()



plt.show()
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)