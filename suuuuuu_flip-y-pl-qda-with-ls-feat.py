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



    skf = StratifiedKFold(n_splits=25, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        qda = QuadraticDiscriminantAnalysis()

        clf = GridSearchCV(qda, params, cv=4)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        reg_params[i] = clf.best_params_['reg_param']

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

    

    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train4p, train2p['target']):

        test_index3 = test_index[ test_index<len(train4) ]

        clf = LabelSpreading(gamma=0.01,kernel='rbf', max_iter=10,n_jobs=-1)

        clf.fit(train4p[train_index,:],train2p.loc[train_index]['target'])

        oof_ls[idx1[test_index3]] = clf.predict_proba(train4[test_index3,:])[:,1]

        pred_te_ls[test2.index] += clf.predict_proba(test4)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'],oof_ls)

print('CV for LabelSpreading =',round(auc,5))  
# STEP 5

for itr in range(4):

    test['target'] = preds

    test.loc[test['target'] > 0.955, 'target'] = 1 # initial 94

    test.loc[test['target'] < 0.045, 'target'] = 0 # initial 06

    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

    usefull_pred_te_ls = pred_te_ls[(test['target'] == 1) | (test['target'] == 0)]

    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)

    new_oof_ls =  np.hstack([oof_ls, usefull_pred_te_ls])

    print(usefull_test.shape[0], "Test Records added for iteration : ", itr)

    new_train.loc[oof > 0.995, 'target'] = 1 # initial 98

    new_train.loc[oof < 0.005, 'target'] = 0 # initial 02

    oof2 = np.zeros(len(train))

    preds = np.zeros(len(test))

    for i in tqdm_notebook(range(512)):



        train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]

        test2 = test[test['wheezy-copper-turtle-magic']==i]

        idx0 = new_train[new_train['wheezy-copper-turtle-magic']==i].index

        idx1 = train[train['wheezy-copper-turtle-magic']==i].index

        idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])

        data2 = pipe.fit_transform(data[cols])

        train3 = data2[:train2.shape[0]]

        test3 = data2[train2.shape[0]:]

        train4 = np.hstack([train3,np.array([new_oof_ls[idx0]]).T])

        test4 = np.hstack([test3,np.array([pred_te_ls[idx2]]).T])

        skf = StratifiedKFold(n_splits=25, random_state=time.time)

        for train_index, test_index in skf.split(train2, train2['target']):

            oof_test_index = [t for t in test_index if t < len(idx1)]

            

            clf = QuadraticDiscriminantAnalysis(reg_params[i])

            clf.fit(train4[train_index,:],train2.loc[train_index]['target'])

            if len(oof_test_index) > 0:

                oof2[idx1[oof_test_index]] = clf.predict_proba(train4[oof_test_index,:])[:,1]

            preds[idx2] += clf.predict_proba(test4)[:,1] / skf.n_splits

    auc = roc_auc_score(train['target'], oof2)

    print(f'AUC: {auc:.5}')

    

# STEP 6

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)
import matplotlib.pyplot as plt

plt.hist(preds,bins=100)

plt.title('Final Test.csv predictions')

plt.show()