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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.info()
test.info()
train.head()
train.set_index("ID_code",inplace=True)

test.set_index("ID_code",inplace=True)



target=train.target

train=train.drop("target",axis=1)
train.describe()
col_counts=dict()

for col in train.columns:

    col_counts[col]=train[col].value_counts().count()

col_counts.values()

# to see if there are some categorical features.

for k,v in col_counts.items():

    if v<50:

        print(k,v)
start=0

features=train.columns.values

rows=8

cols=8

fig,axs=plt.subplots(rows,cols,figsize=(18,18))

for row in range(rows):

    for col in range(cols):

        feature=features[row*rows+col]

        ax=axs[row][col]

        sns.distplot(train[feature],bins=20,ax=ax)
from sklearn.linear_model import LogisticRegression

import time

from sklearn import metrics

from sklearn.model_selection import KFold
model=LogisticRegression(C=1,n_jobs=4,penalty="l2")
kford=KFold(n_splits=5,random_state=2,shuffle=True)

start_time=time.time()

aucs=[]



test_preds=[]

# for early stopping

# it takes a long time if using all the samples.

samples=train.shape[0]//20

for ford,(train_idx,val_idx) in enumerate(kford.split(train[:samples],target[:samples])):

    print("####################################")

    print("############ford:",ford)

    sample_x=train.iloc[train_idx].values

    sample_y=target.iloc[train_idx].values

    

    sample_val_x=train.iloc[val_idx].values

    sample_val_y=target.iloc[val_idx].values

    

    model.fit(sample_x,sample_y)

    y_pred_prob=model.predict_proba(sample_x)[:,1]

    y_val_pred_prob=model.predict_proba(sample_val_x)[:,1]

    

    train_auc=metrics.roc_auc_score(sample_y,y_pred_prob)

    val_auc=metrics.roc_auc_score(sample_val_y,y_val_pred_prob)

    print("train auc:{},val auc:{}".format(train_auc,val_auc))

    aucs.append([train_auc,val_auc])

    test_preds.append(model.predict_proba(test)[:,1])

    

end_time=time.time()

val_aucs=[auc[1] for auc in aucs]

print("using {} samples,total time:{}s,mean val auc:{}".format(samples,end_time-start_time,np.mean(val_aucs)))
test_preds=pd.DataFrame(test_preds).T

test_preds.index=test.index
test_preds.head()
submission=test_preds.mean(axis=1)

submission=pd.DataFrame(submission)

submission.columns=["target"]

submission.head()

submission.to_csv("submission.csv")
!head -5 "submission.csv"