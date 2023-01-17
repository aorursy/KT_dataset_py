# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv",index_col="ID_code")

test=pd.read_csv("../input/test.csv",index_col="ID_code")

target=train.target

train=train.drop("target",axis=1)
train.head()
print("all the features are numeric ,and the distribution are of gaussian")
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.model_selection import KFold

import time
def train_model(model,train,target,test,samples=-1,**params):    

    kford=KFold(n_splits=5,random_state=2,shuffle=True)

    start_time=time.time()

    aucs=[]



    test_preds=[]

    # for early stopping

    # it takes a long time if using all the samples.

    if samples<=-1:

        samples=train.shape[0]

    else:

        samples=min(train.shape[0],samples)

    print("##################################################################")

    print("########## start fit model ###################")

    print("fit on {} samples".format(samples))

    for ford,(train_idx,val_idx) in enumerate(kford.split(train[:samples],target[:samples])):

        print("####################################")

        print("############ford:",ford)

        sample_x=train.iloc[train_idx].values

        sample_y=target.iloc[train_idx].values



        sample_val_x=train.iloc[val_idx].values

        sample_val_y=target.iloc[val_idx].values

        

        ford_time=time.time()

        model.fit(sample_x,sample_y)

        print("epoch cost time {:1}s".format(time.time()-ford_time))

        y_pred_prob=model.predict_proba(sample_x)[:,1]

        y_val_pred_prob=model.predict_proba(sample_val_x)[:,1]



        train_auc=metrics.roc_auc_score(sample_y,y_pred_prob)

        val_auc=metrics.roc_auc_score(sample_val_y,y_val_pred_prob)

        print("train auc:{:4},val auc:{:4}".format(train_auc,val_auc))

        aucs.append([train_auc,val_auc])

        test_preds.append(model.predict_proba(test)[:,1])



    end_time=time.time()

    val_aucs=[auc[1] for auc in aucs]

    print("using {} samples,total time:{:1}s,mean val auc:{:4}".format(samples,end_time-start_time,np.mean(val_aucs)))

    test_preds=pd.DataFrame(test_preds).T

    test_preds.index=test.index

    return test_preds

model=GaussianNB()

test_preds=train_model(model,train,target,test,samples=-1)
submission=pd.DataFrame(test_preds.mean(axis=1),columns=["target"])
submission.head()
submission.to_csv("submission.csv")
!head -5 "submission.csv"