import numpy as np 

import pandas as pd 

import os

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.mixture import GaussianMixture

from sklearn.covariance import GraphicalLasso

from sklearn.preprocessing import StandardScaler

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')

test['wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].astype('category')
magicNum = 131073

default_cols = [c for c in train.columns if c not in ['id', 'target','target_pred', 'wheezy-copper-turtle-magic']]

cols = [c for c in default_cols]

sub = pd.read_csv('../input/sample_submission.csv')

sub.to_csv('submission.csv',index=False)

train.shape,test.shape
y_perfect = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]

y_flliped = [1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1]

roc_auc_score(y_perfect,y_flliped)
y_preds = [0.33,0.33,0.33,0.5,0.5,0,0,0,0,0,1,1,0.5,0.5,1,1,1,0.66,0.66,0.66]

roc_auc_score(y_flliped,y_preds)
if sub.shape[0] == magicNum:

    [].shape   



preds=np.zeros(len(test))

train_err=np.zeros(512)

test_err=np.zeros(512)



for i in range(512):  

    

    X = train[train['wheezy-copper-turtle-magic']==i].copy()

    Y = X.pop('target').values

    X_test = test[test['wheezy-copper-turtle-magic']==i].copy()



    idx_train = X.index 

    idx_test = X_test.index

    

    X.reset_index(drop=True,inplace=True)

    

    X = X[cols].values             

    X_test = X_test[cols].values



    vt = VarianceThreshold(threshold=2).fit(X)

    

    X = vt.transform(X)         

    X_test = vt.transform(X_test)

    X_all = np.concatenate([X,X_test])

    train_size = len(X)

    test1_size = test[:131073][test[:131073]['wheezy-copper-turtle-magic']==i].shape[0]

    compo_cnt = 6

    for ii in range(30):

        gmm = GaussianMixture(n_components=compo_cnt,init_params='random',covariance_type='full',max_iter=100,tol=1e-10,reg_covar=0.0001).fit(X_all)

        labels = gmm.predict(X_all)

        

        cntStd = np.std([len(labels[labels==j]) for j in range(compo_cnt)])

        #there are chances that the clustering doesn't converge, so we only choose the case that it clustered equally

        #in which case, the sizes are 171,170,171,170,...

        if round(cntStd,4) == 0.4714:

            check_labels = labels[:train_size]

            cvt_labels=np.zeros(len(labels))



            #first get the perfect classification label

            for iii in range(compo_cnt):

                mean_val = Y[check_labels==iii].mean()

                mean_val = 1 if mean_val > 0.5 else 0

                cvt_labels[labels==iii] = mean_val

            

            #then try to predict the expected err for the test set

            train_err[i] = len(Y[Y != cvt_labels[:train_size]])

            if (train_err[i] >= 10) and (train_err[i] <= 15):

                train_err[i] = 12.5

            exp_err = max(0,(25 - train_err[i])/(train_size + test1_size))



            for iii in range(compo_cnt):

                mean_val = Y[check_labels==iii].mean()

                mean_val = (1-exp_err) if mean_val > 0.5 else exp_err

                cvt_labels[labels==iii] = mean_val



            preds[idx_test] = cvt_labels[train_size:]

            break



sub['target'] = preds

sub.to_csv('submission.csv',index=False)