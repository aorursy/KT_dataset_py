import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from tqdm import tqdm_notebook

import warnings

import multiprocessing

from scipy.optimize import minimize  

import time

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.semi_supervised import LabelSpreading

warnings.filterwarnings('ignore')
# STEP 2

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

print(train.shape, test.shape)
# STEP 3

oof = np.zeros(len(train))

preds = np.zeros(len(test))

params = [{'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}]

# 512 models

reg_params = np.zeros(512)

for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])

    data2 = pipe.fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



#         qda = QuadraticDiscriminantAnalysis()

#         clf = GridSearchCV(qda, params, cv=4)

        clf = QuadraticDiscriminantAnalysis(0.25)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

#         reg_params[i] = clf.best_params_['reg_param']

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')
# STEP 4

train_f = train.copy()

train_f.loc[oof > 0.995, 'target'] = 1

train_f.loc[oof < 0.005, 'target'] = 0

oof_ls = np.zeros(len(train)) 

pred_te_ls = np.zeros(len(test))

for k in tqdm_notebook(range(512)):

    train2 = train_f[train_f['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    test2['target']=-1

    train2p = pd.concat([train2,test2],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train4p = sel.transform(train2p[cols])

    train4 = sel.transform(train2[cols])

    test4 = sel.transform(test2[cols])

    

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train4p, train2p['target']):

        test_index3 = test_index[ test_index<len(train4) ]

        clf = LabelSpreading(gamma=0.01,kernel='rbf', max_iter=10,n_jobs=-1)

        clf.fit(train4p[train_index,:],train2p.loc[train_index]['target'])

        oof_ls[idx1[test_index3]] = clf.predict_proba(train4[test_index3,:])[:,1]

        pred_te_ls[test2.index] += clf.predict_proba(test4)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'],oof_ls)

print('CV for LabelSpreading =',round(auc,5))  
from sklearn.covariance import GraphicalLasso



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
from sklearn.mixture import GaussianMixture



# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in tqdm_notebook(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

    train4 = np.hstack([train3,np.array([oof_ls[idx1]]).T])

    test4 = np.hstack([test3,np.array([pred_te_ls[idx2]]).T])    

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train4, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        ms, ps = get_mean_cov(train4[train_index,:],train2.loc[train_index]['target'].values)

        

        gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)

        gm.fit(np.concatenate([train4[train_index,:],test4],axis = 0))

        oof[idx1[test_index]] = gm.predict_proba(train4[test_index,:])[:,0]

        preds[idx2] += gm.predict_proba(test4)[:,0] / skf.n_splits



        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('QDA scores CV =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)



import matplotlib.pyplot as plt

plt.hist(preds,bins=100)

plt.title('Final Test.csv predictions')

plt.show()