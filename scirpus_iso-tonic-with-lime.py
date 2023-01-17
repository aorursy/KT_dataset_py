import numpy as np, pandas as pd, os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from matplotlib import pyplot as plt

from  tqdm import tqdm

from sklearn.isotonic import IsotonicRegression

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
# INITIALIZE VARIABLES



cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))

preds = np.zeros(len(test))

isooof = np.zeros(len(train))

isopreds = np.zeros(len(test))

keys = []

scores = []

thresholds = []

regs = []

# BUILD 512 SEPARATE MODELS

for i in tqdm(range(512)):

    keys.append(i)

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    bestscore = 0

    bestthreshold = 2.

    bestreg = .3

    for t in np.arange(2,0.9,-.1):

        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

        sel = VarianceThreshold(threshold=t).fit(train2[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])



        # STRATIFIED K-FOLD

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

        for r in np.arange(0.3,0.6,0.1):

            indscores = []

            for train_index, test_index in skf.split(train3, train2['target']):

                clf = QuadraticDiscriminantAnalysis(reg_param=r)



                clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

                score = roc_auc_score(train2.loc[test_index]['target'],clf.predict_proba(train3[test_index,:])[:,1])

                indscores.append(score)

            #print(np.mean(indscores))

            if(np.mean(indscores)>bestscore):

                bestscore=np.mean(indscores)

                bestthreshold = t

                bestreg = r

    

    regs.append(bestreg)

    thresholds.append(bestthreshold)

    scores.append(bestscore)

    sel = VarianceThreshold(threshold=bestthreshold).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])



    for i, (train_index, test_index) in enumerate(skf.split(train3, train2['target'])):



        # MODEL AND PREDICT WITH QDA



        clf = QuadraticDiscriminantAnalysis(reg_param=bestreg)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        indscores.append(roc_auc_score(train2.loc[test_index]['target'],oof[idx1[test_index]]) )

        

    iso = IsotonicRegression(y_min=0, y_max=1)

    iso.fit(oof[idx1],train2['target'].values)

    isooof[idx1] = iso.transform(oof[idx1])

    isopreds[idx2] = iso.transform(preds[idx2])

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)





print('QDA scores CV =',round(auc,5))



auc = roc_auc_score(train['target'],isooof)



print('QDA scores CV =',round(auc,5))
sub = pd.DataFrame()

sub['id'] = test.id.values

sub['target'] = preds

sub.to_csv('qdasubmission.csv',index=False)

sub = pd.DataFrame()

sub['id'] = test.id.values

sub['target'] = isopreds

sub.to_csv('isoqdasubmission.csv',index=False)
from sklearn.mixture import GaussianMixture

from sklearn.covariance import GraphicalLasso

import warnings

warnings.simplefilter("ignore")

def get_mean_cov(x,y):

    model = GraphicalLasso()

    ones = (y==1).astype(bool)

    x2 = x[ones]

    model.fit(x2)

    p1 = model.precision_

    m1 = model.location_

    

    onesb = (y==0).astype(bool)

    x2b = x[onesb]

    model.fit(x2b)

    p2 = model.precision_

    m2 = model.location_

    

    ms = np.stack([m1,m2])

    ps = np.stack([p1,p2])

    return ms,ps





# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))

preds = np.zeros(len(test))

isooof = np.zeros(len(train))

isopreds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS

keys = []

scores = []

thresholds = []

regs = []

for i in tqdm(range(512)):

    keys.append(i)

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    bestscore = 0

    bestthreshold = 2.

    bestreg = 100

    for t in np.arange(1.6,1.3,-.1):

        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

        sel = VarianceThreshold(threshold=t).fit(train2[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])



        # STRATIFIED K-FOLD

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

        for r in np.arange(100,400,100):

            indscores = []

            for train_index, test_index in skf.split(train3, train2['target']):

                ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)

        

                clf = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001,reg_covar=.001, max_iter=r, n_init=1,means_init=ms, precisions_init=ps)

                clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

                score = roc_auc_score(train2.loc[test_index]['target'],clf.predict_proba(train3[test_index,:])[:,1])

                indscores.append(score)

            #print(np.mean(indscores))

            if(np.mean(indscores)>bestscore):

                bestscore=np.mean(indscores)

                bestthreshold = t

                bestreg = r

    

    regs.append(bestreg)

    thresholds.append(bestthreshold)

    scores.append(bestscore)

    sel = VarianceThreshold(threshold=bestthreshold).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])



        

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)

        

        gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=bestreg, n_init=1,means_init=ms, precisions_init=ps)

        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))

        

        oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:,0]

        preds[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits

    iso = IsotonicRegression(y_min=0, y_max=1)

    iso.fit(oof[idx1],train2['target'].values)

    isooof[idx1] = iso.transform(oof[idx1])

    isopreds[idx2] = iso.transform(preds[idx2])

        

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)



print('QDA scores CV =',round(auc,5))



auc = roc_auc_score(train['target'],isooof)



print('QDA scores CV =',round(auc,5))
sub = pd.DataFrame()

sub['id'] = test.id.values

sub['target'] = preds

sub.to_csv('gmmsubmission.csv',index=False)

sub = pd.DataFrame()

sub['id'] = test.id.values

sub['target'] = isopreds

sub.to_csv('isogmmsubmission.csv',index=False)