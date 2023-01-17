import numpy as np, pandas as pd, os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook as tqdm

from sklearn.covariance import LedoitWolf, OAS

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import RobustScaler

from sklearn.cluster import KMeans, Birch

from sklearn.pipeline import Pipeline

from scipy.stats import multivariate_normal

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import rankdata

from scipy.optimize import minimize

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof    = np.zeros(len(train))

preds  = np.zeros(len(test))

oof2   = np.zeros(len(train))

preds2 = np.zeros(len(test))

oof3   = np.zeros(len(train))

preds3 = np.zeros(len(test))

ooff   = np.zeros(len(train))

predsf = np.zeros(len(test))

K2=[]

K3=[]

K4=[]

Seed = 777



# BUILD 512 SEPARATE MODELS

for i in tqdm(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    #pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler',  RobustScaler(quantile_range=(25, 75)))])

    pipe = Pipeline([('vt', VarianceThreshold(threshold=1.5)), ('scaler',  RobustScaler(quantile_range=(35, 65)))])

    sel = pipe.fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])



    NK = 2

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        oof_test_index = [t for t in test_index if t < len(idx1)]



        x = train3[train_index]

        y = train2.loc[train_index,'target'].values

        x1 = x[(y==1).astype(bool)]

        cc1 = GaussianMixture(n_components=NK, init_params='random',covariance_type='full', n_init=1,random_state=Seed, reg_covar=0.075).fit_predict(x1)

        x11 = x1[cc1==0]

        x12 = x1[cc1==1]

        model11 = OAS(assume_centered =False).fit(x11)

        p11 = model11.precision_

        m11 = model11.location_ 

        model12 = OAS(assume_centered =False).fit(x12)

        p12 = model12.precision_

        m12 = model12.location_ 



        x2 = x[(y==0).astype(bool)]

        cc2 = GaussianMixture(n_components=NK, init_params='random', covariance_type='full',

                              n_init=1,random_state=Seed, reg_covar=0.075).fit_predict(x2)

        x21 = x2[cc2==0]

        x22 = x2[cc2==1]

        model21 =  OAS(assume_centered =False).fit(x21)

        p21 = model21.precision_

        m21 = model21.location_ 

        model22 =  OAS(assume_centered =False).fit(x22)

        p22 = model22.precision_

        m22 = model22.location_ 



        ms = np.stack([m11,m12,m21,m22])

        ps = np.stack([p11,p12,p21,p22])



        gm2 = GaussianMixture(n_components=4,random_state=42, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=1000 )

        gm2.fit(np.concatenate([train3[train_index],test3],axis = 0))

        oof[idx1[oof_test_index]] = gm2.predict_proba(train3[oof_test_index,:])[:,0:NK].sum(axis=1)

        preds[idx2] += gm2.predict_proba(test3)[:,0:NK].sum(axis=1) / skf.n_splits

    sc2 = roc_auc_score(train2['target'], oof[idx1] )





    

    NK = 3

    # STRATIFIED K-FOLD

    for train_index, test_index in skf.split(train3, train2['target']):

        oof_test_index = [t for t in test_index if t < len(idx1)]



        x = train3[train_index]

        y = train2.loc[train_index,'target'].values

        x1 = x[(y==1).astype(bool)]

        cc1 = GaussianMixture(n_components=NK, init_params='random',covariance_type='full',

                              n_init=1,random_state=Seed, reg_covar=0.075).fit_predict(x1)

        x11 = x1[cc1==0]

        x12 = x1[cc1==1]

        x13 = x1[cc1==2]

        model11 = OAS(assume_centered =False).fit(x11)

        p11 = model11.precision_

        m11 = model11.location_ 

        model12 = OAS(assume_centered =False).fit(x12)

        p12 = model12.precision_

        m12 = model12.location_ 

        model13 = OAS(assume_centered =False).fit(x13)

        p13 = model13.precision_

        m13 = model13.location_ 



        x2 = x[(y==0).astype(bool)]

        cc2 = GaussianMixture(n_components=NK, init_params='random', covariance_type='full',

                              n_init=1,random_state=Seed, reg_covar=0.075).fit_predict(x2)

        x21 = x2[cc2==0]

        x22 = x2[cc2==1]

        x23 = x2[cc2==2]

        model21 =  OAS(assume_centered =False).fit(x21)

        p21 = model21.precision_

        m21 = model21.location_ 

        model22 =  OAS(assume_centered =False).fit(x22)

        p22 = model22.precision_

        m22 = model22.location_ 

        model23 =  OAS(assume_centered =False).fit(x23)

        p23 = model23.precision_

        m23 = model23.location_



        ms = np.stack([m11,m12,m13,m21,m22,m23])

        ps = np.stack([p11,p12,p13,p21,p22,p23])



        gm3 = GaussianMixture(n_components=6,random_state=42, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=1000 )

        gm3.fit(np.concatenate([train3[train_index],test3],axis = 0))

        oof2[idx1[oof_test_index]] = gm3.predict_proba(train3[oof_test_index,:])[:,0:NK].sum(axis=1)

        preds2[idx2] += gm3.predict_proba(test3)[:,0:NK].sum(axis=1) / skf.n_splits

    sc3 = roc_auc_score(train2['target'], oof2[idx1])



    NK = 4

    # STRATIFIED K-FOLD

    for train_index, test_index in skf.split(train3, train2['target']):

        oof_test_index = [t for t in test_index if t < len(idx1)]



        x = train3[train_index]

        y = train2.loc[train_index,'target'].values

        x1 = x[(y==1).astype(bool)]

        cc1 = GaussianMixture(n_components=NK, init_params='random', covariance_type='full',

                              n_init=1,random_state=Seed, reg_covar=0.075).fit_predict(x1)

        x11 = x1[cc1==0]

        x12 = x1[cc1==1]

        x13 = x1[cc1==2]

        x14 = x1[cc1==3]

        model11 = OAS(assume_centered =False).fit(x11)

        p11 = model11.precision_

        m11 = model11.location_ 

        model12 = OAS(assume_centered =False).fit(x12)

        p12 = model12.precision_

        m12 = model12.location_ 

        model13 = OAS(assume_centered =False).fit(x11)

        p13 = model13.precision_

        m13 = model13.location_ 

        model14 = OAS(assume_centered =False).fit(x14)

        p14 = model14.precision_

        m14 = model14.location_ 



        x2 = x[(y==0).astype(bool)]

        cc2 = GaussianMixture(n_components=NK, init_params='random', covariance_type='full',

                              n_init=1,random_state=Seed, reg_covar=0.075).fit_predict(x2)

        x21 = x2[cc2==0]

        x22 = x2[cc2==1]

        x23 = x2[cc2==2]

        x24 = x2[cc2==3]

        model21 =  OAS(assume_centered =False).fit(x21)

        p21 = model21.precision_

        m21 = model21.location_ 

        model22 =  OAS(assume_centered =False).fit(x22)

        p22 = model22.precision_

        m22 = model22.location_ 

        model23 =  OAS(assume_centered =False).fit(x21)

        p23 = model23.precision_

        m23 = model23.location_ 

        model24 =  OAS(assume_centered =False).fit(x24)

        p24 = model24.precision_

        m24 = model24.location_



        ms = np.stack([m11,m12,m13,m14,m21,m22,m23,m24])

        ps = np.stack([p11,p12,p13,p14,p21,p22,p23,p24])



        gm4 = GaussianMixture(n_components=8,random_state=42, covariance_type='full', means_init=ms, precisions_init=ps, tol=0.000001, max_iter=1000 )

        gm4.fit(np.concatenate([train3[train_index],test3],axis = 0))

        oof3[idx1[oof_test_index]] = gm4.predict_proba(train3[oof_test_index,:])[:,0:NK].sum(axis=1)

        preds3[idx2] += gm4.predict_proba(test3)[:,0:NK].sum(axis=1) / skf.n_splits

    sc4 = roc_auc_score(train2['target'], oof3[idx1])



    scmax = max(sc2, sc3, sc4)

    

    if scmax == sc2:

        

        ooff[idx1]=oof[idx1]

        predsf[idx2]=preds[idx2]

    

    elif scmax == sc3:

       

        ooff[idx1]=oof2[idx1]

        predsf[idx2]=preds2[idx2]

        

    elif scmax == sc4:

        ooff[idx1]=oof2[idx1]

        predsf[idx2]=preds3[idx2]

        

    K2.append(sc2)

    K3.append(sc3)

    K4.append(sc4)



    print( sc2,sc3,sc4 )

    

    

    

    
print(roc_auc_score(train['target'], ooff))
print(K2)

print(K3)

print(K4)
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = predsf

sub.to_csv('submission.csv',index=False)