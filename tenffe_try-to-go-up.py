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

warnings.filterwarnings('ignore')
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

print(train.shape, test.shape)
reg_params = [0.01,0.23,0.01,0.08,0.27,0.03,0.16,0.33,0.28,0.02,0.01,0.07,0.01,0.31

,0.28,0.26,0.02,0.1,0.11,0.17,0.42,0.34,0.11,0.22,0.61,0.12,0.14,0.19

,0.02,0.01,0.02,0.06,0.36,0.09,0.11,0.01,0.03,0.11,0.01,0.11,0.23,0.02

,0.17,0.14,0.12,0.06,0.08,0.01,0.26,0.01,0.02,0.04,0.33,0.03,0.07,0.16

,0.01,0.02,0.28,0.17,0.04,0.19,0.05,0.11,0.03,0.22,0.17,0.16,0.06,0.14

,0.01,0.07,0.05,0.06,0.2,0.05,0.02,0.09,0.05,0.01,0.08,0.1,0.01,0.22

,0.19,0.13,0.02,0.03,0.12,0.07,0.15,0.09,0.02,0.16,0.01,0.39,0.12,0.14

,0.09,0.41,0.32,0.13,0.15,0.01,0.41,0.12,0.01,0.13,0.17,0.03,0.08,0.12

,0.21,0.01,0.16,0.28,0.01,0.02,0.11,0.21,0.27,0.03,0.17,0.05,0.02,0.4

,0.28,0.01,0.03,0.28,0.02,0.12,0.17,0.1,0.01,0.1,0.01,0.24,0.11,0.18

,0.17,0.08,0.1,0.04,0.13,0.04,0.12,0.25,0.13,0.19,0.05,0.03,0.1,0.01

,0.05,0.01,0.01,0.01,0.18,0.24,0.13,0.01,0.15,0.01,0.09,0.04,0.06,0.01

,0.06,0.11,0.21,0.08,0.21,0.04,0.04,0.04,0.2,0.01,0.19,0.14,0.11,0.03

,0.23,0.01,0.04,0.13,0.46,0.04,0.07,0.08,0.07,0.01,0.11,0.2,0.07,0.23

,0.2,0.14,0.07,0.06,0.16,0.02,0.33,0.13,0.11,0.06,0.22,0.22,0.19,0.03

,0.1,0.37,0.22,0.01,0.01,0.22,0.07,0.01,0.23,0.35,0.03,0.29,0.01,0.04

,0.01,0.04,0.07,0.23,0.2,0.09,0.01,0.23,0.3,0.06,0.09,0.04,0.46,0.25

,0.14,0.01,0.04,0.03,0.01,0.04,0.11,0.08,0.01,0.09,0.05,0.1,0.05,0.28

,0.02,0.08,0.01,0.06,0.38,0.04,0.01,0.15,0.21,0.01,0.01,0.45,0.18,0.27

,0.24,0.01,0.04,0.14,0.13,0.18,0.22,0.32,0.13,0.07,0.26,0.17,0.12,0.14

,0.09,0.13,0.08,0.09,0.07,0.01,0.02,0.04,0.01,0.07,0.32,0.01,0.36,0.09

,0.11,0.06,0.46,0.11,0.16,0.21,0.01,0.1,0.01,0.1,0.23,0.05,0.33,0.01

,0.24,0.04,0.01,0.04,0.1,0.01,0.36,0.44,0.03,0.08,0.21,0.01,0.18,0.01

,0.17,0.19,0.03,0.01,0.18,0.15,0.48,0.06,0.17,0.18,0.37,0.01,0.31,0.01

,0.16,0.18,0.11,0.08,0.08,0.07,0.28,0.02,0.09,0.08,0.01,0.09,0.01,0.07

,0.01,0.24,0.09,0.02,0.37,0.16,0.04,0.14,0.22,0.06,0.29,0.16,0.06,0.06

,0.04,0.05,0.25,0.07,0.01,0.01,0.21,0.02,0.04,0.3,0.39,0.02,0.23,0.22

,0.05,0.01,0.06,0.05,0.02,0.01,0.02,0.12,0.11,0.05,0.01,0.2,0.01,0.08

,0.08,0.04,0.33,0.06,0.16,0.35,0.18,0.13,0.01,0.01,0.12,0.18,0.01,0.01

,0.17,0.08,0.48,0.18,0.01,0.02,0.35,0.15,0.34,0.01,0.14,0.01,0.32,0.34

,0.15,0.1,0.18,0.18,0.11,0.24,0.01,0.13,0.03,0.36,0.01,0.08,0.01,0.13

,0.07,0.08,0.24,0.01,0.05,0.05,0.1,0.07,0.21,0.01,0.08,0.11,0.09,0.04

,0.01,0.11,0.02,0.09,0.16,0.51,0.17,0.09,0.03,0.12,0.06,0.18,0.01,0.01

,0.02,0.02,0.01,0.27,0.28,0.09,0.02,0.13,0.03,0.16,0.15,0.04,0.17,0.19

,0.26,0.01,0.04,0.1,0.04,0.01,0.07,0.05,0.02,0.59,0.01,0.2,0.12,0.1

,0.07,0.18,0.08,0.01,0.34,0.15,0.28,0.12,0.05,0.01,0.11,0.08,0.12,0.04

,0.05,0.33,0.01,0.01,0.09,0.07,0.19,0.09]
def QDA_train(train_qda, test_qda):

    oof = np.zeros(len(train_qda))

    preds = np.zeros(len(test_qda))



    for i in tqdm_notebook(range(512)):



        train2 = train_qda[train_qda['wheezy-copper-turtle-magic']==i]

        test2 = test_qda[test_qda['wheezy-copper-turtle-magic']==i]

        idx1 = train2.index; idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])

        data2 = pipe.fit_transform(data[cols])

        train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



        skf = StratifiedKFold(n_splits=31, random_state=42)

        for train_index, test_index in skf.split(train2, train2['target']):



            clf = QuadraticDiscriminantAnalysis(reg_params[i])

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



    auc = roc_auc_score(train_qda['target'], oof)

    print(f'AUC: {auc:.5}')

    return oof, preds
oof_qda, preds_qda = QDA_train(train, test)
LOW = 0.45

HIGH = 1 - LOW
def get_denoise_data(train_off, oof, low, high):

    print(len(train_off[(oof > low) & (oof < high)]))

    train_off_filter = train_off[(oof <= low) | (oof >= high)]

    train_off_filter.reset_index(drop=True, inplace=True)

    return train_off_filter
train_outlier = train.copy()

oof_outlier = oof_qda.copy()
NUM_DENOISE = 2

for i in range(NUM_DENOISE):

    train_outlier = get_denoise_data(train_outlier, oof_outlier, LOW, HIGH)

    oof_outlier, preds_outlier = QDA_train(train_outlier, test)
def Pseudo_train(preds, oof, train_pse, test_pse):

    for itr in range(2):

        test_pse['target'] = preds

        test_pse.loc[test_pse['target'] > 0.955, 'target'] = 1 # initial 94

        test_pse.loc[test_pse['target'] < 0.045, 'target'] = 0 # initial 06

        usefull_test = test_pse[(test_pse['target'] == 1) | (test_pse['target'] == 0)]

        new_train = pd.concat([train_pse, usefull_test]).reset_index(drop=True)

        print(usefull_test.shape[0], "Test Records added for iteration : ", itr)

        new_train.loc[oof > 0.995, 'target'] = 1 # initial 98

        new_train.loc[oof < 0.005, 'target'] = 0 # initial 02

        oof = np.zeros(len(train_pse))

        preds = np.zeros(len(test_pse))

        for i in tqdm_notebook(range(512)):



            train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]

            test2 = test[test['wheezy-copper-turtle-magic']==i]

            idx1 = train_pse[train_pse['wheezy-copper-turtle-magic']==i].index

            idx2 = test2.index

            train2.reset_index(drop=True,inplace=True)



            data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

            pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])

            data2 = pipe.fit_transform(data[cols])

            train3 = data2[:train2.shape[0]]

            test3 = data2[train2.shape[0]:]



            skf = StratifiedKFold(n_splits=31, random_state=time.time)

            for train_index, test_index in skf.split(train2, train2['target']):

                oof_test_index = [t for t in test_index if t < len(idx1)]



                clf = QuadraticDiscriminantAnalysis(reg_params[i])

                clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

                if len(oof_test_index) > 0:

                    oof[idx1[oof_test_index]] = clf.predict_proba(train3[oof_test_index,:])[:,1]

                preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        auc = roc_auc_score(train_pse['target'], oof)

        print(f'AUC: {auc:.5}')

        

        return oof, preds
oof_pse, preds_pse = Pseudo_train(preds_outlier, oof_outlier, train_outlier, test)
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



def GMM_train(train_GMM, test_GMM):

    # INITIALIZE VARIABLES

    cols = [c for c in train_GMM.columns if c not in ['id', 'target']]

    cols.remove('wheezy-copper-turtle-magic')

    oof_GMM = np.zeros(len(train_GMM))

    preds_GMM = np.zeros(len(test_GMM))



    # BUILD 512 SEPARATE MODELS

    for i in tqdm_notebook(range(512)):

        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

        train2 = train_GMM[train_GMM['wheezy-copper-turtle-magic']==i]

        test2 = test_GMM[test_GMM['wheezy-copper-turtle-magic']==i]

        idx1 = train2.index; idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])



        # STRATIFIED K-FOLD

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

        for train_index, test_index in skf.split(train3, train2['target']):



            # MODEL AND PREDICT WITH QDA

            ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)



            gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)

            gm.fit(np.concatenate([train3,test3],axis = 0))

            oof_GMM[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:,0]

            preds_GMM[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits





    # PRINT CV AUC

    auc = roc_auc_score(train_GMM['target'],oof_GMM)

    print('QDA scores CV =',round(auc,5))

    return oof_GMM, preds_GMM
oof_GMM, preds_GMM = GMM_train(train_outlier, test)
preds = preds_pse * 0.8 + preds_GMM * 0.2
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds_pse
sub.to_csv('submission.csv',index=False)