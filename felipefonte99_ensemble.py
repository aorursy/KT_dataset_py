import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from tqdm import tqdm_notebook

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.mixture import GaussianMixture

import warnings

warnings.filterwarnings('ignore')
n_folds = 5

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

print(train.shape, test.shape)
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
oof = np.zeros(len(train))



for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])

    data2 = pipe.fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=n_folds, random_state=42)

    for train_index, val_index in skf.split(train2, train2['target']):



        clf = QuadraticDiscriminantAnalysis(0.5)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[val_index]] = clf.predict_proba(train3[val_index,:])[:,1]



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')
new_train = train.copy()

new_train.loc[oof > 0.98, 'target'] = 1

new_train.loc[oof < 0.02, 'target'] = 0
oof_gm = np.zeros(len(train))

oof_bc_qda = np.zeros(len(train))

oof_bc_knn = np.zeros(len(train))

oof_svc = np.zeros(len(train))



preds_gm = np.zeros(len(test))

preds_bc_qda = np.zeros(len(test))

preds_bc_knn = np.zeros(len(test))

preds_svc = np.zeros(len(test))



for i in tqdm_notebook(range(512)):



    train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])

    data2 = pipe.fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=n_folds, random_state=42)

    for train_index, val_index in skf.split(train2, train2['target']):



        clf_qda = QuadraticDiscriminantAnalysis(0.5)        

        clf = BaggingClassifier(base_estimator=clf_qda, n_estimators=40, n_jobs=-1, random_state=42)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_bc_qda[idx1[val_index]] = clf.predict_proba(train3[val_index,:])[:,1]

        preds_bc_qda[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        clf_knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

        clf = BaggingClassifier(base_estimator=clf_knn, n_estimators=40, n_jobs=-1, random_state=42)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_bc_knn[idx1[val_index]] = clf.predict_proba(train3[val_index,:])[:,1]

        preds_bc_knn[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



        clf_svc = SVC(random_state=42, probability=True)

        clf_svc.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_svc[idx1[val_index]] = clf_svc.predict_proba(train3[val_index,:])[:,1]

        preds_svc[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



        ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)

        gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001, reg_covar=0.001, max_iter=100, n_init=1, means_init=ms, precisions_init=ps)

        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))

        oof_gm[idx1[val_index]] = gm.predict_proba(train3[val_index,:])[:,0]

        oof_gm[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits



auc = roc_auc_score(train['target'], oof_bc_qda)

print(f'AUC bc qda: {auc:.5}')



auc = roc_auc_score(train['target'], oof_svc)

print(f'AUC svc: {auc:.5}')



auc = roc_auc_score(train['target'], oof_bc_knn)

print(f'AUC bc knn: {auc:.5}')



auc = roc_auc_score(train['target'], oof_gm)

print(f'AUC gm: {auc:.5}')
preds_blend1 = 0.68*preds_gm + 0.32*preds_svc

preds_blend = 0.55*preds_bc_qda + 0.25*preds_blend1 + 0.2*preds_bc_knn



oof_blend1 = 0.68*oof_gm + 0.32*oof_svc

oof_blend = 0.55*oof_bc_qda + 0.25*oof_blend1 + 0.2*oof_bc_knn



auc = roc_auc_score(train['target'], oof_blend)

print(f'AUC: {auc:.7}')
for itr in range(2):

    

    test['target'] = preds_blend

    test.loc[test['target'] > 0.60, 'target'] = 1

    test.loc[test['target'] < 0.40, 'target'] = 0

        

    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)

    print(usefull_test.shape[0], "Test Records added for iteration : ", itr)

    oof_gm = np.zeros(len(train))

    oof_bc_qda = np.zeros(len(train))

    oof_bc_knn = np.zeros(len(train))

    oof_svc = np.zeros(len(train))

    preds_gm = np.zeros(len(test))

    preds_bc_qda = np.zeros(len(test))

    preds_knn = np.zeros(len(test))

    preds_svc = np.zeros(len(test))

    

    for i in tqdm_notebook(range(512)):



        train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]

        test2 = test[test['wheezy-copper-turtle-magic']==i]

        idx1 = train[train['wheezy-copper-turtle-magic']==i].index

        idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])

        data2 = pipe.fit_transform(data[cols])

        train3 = data2[:train2.shape[0]]

        test3 = data2[train2.shape[0]:]



        skf = StratifiedKFold(n_splits=n_folds, random_state=42)

        for train_index, val_index in skf.split(train2, train2['target']):

            oof_val_index = [t for t in val_index if t < len(idx1)]

            

            clf_qda = QuadraticDiscriminantAnalysis(0.5)            

            clf = BaggingClassifier(base_estimator=clf_qda, n_estimators=40, n_jobs=-1, random_state=42)

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            if len(oof_val_index) > 0:

                oof_bc_qda[idx1[oof_val_index]] = clf.predict_proba(train3[oof_val_index,:])[:,1]

            preds_bc_qda[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

                

            clf_knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)            

            clf = BaggingClassifier(base_estimator=clf_knn, n_estimators=40, n_jobs=-1, random_state=42)

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            if len(oof_val_index) > 0:

                oof_bc_knn[idx1[oof_val_index]] = clf.predict_proba(train3[oof_val_index,:])[:,1]

            preds_bc_knn[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

                

            clf = SVC(random_state=42, probability=True)

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            if len(oof_val_index) > 0:

                oof_svc[idx1[oof_val_index]] = clf.predict_proba(train3[oof_val_index,:])[:,1]

            preds_svc[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

            

            ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)

            gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001, reg_covar=0.001, max_iter=100, n_init=1, means_init=ms, precisions_init=ps)

            gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))

            if len(oof_val_index) > 0:

                oof_gm[idx1[oof_val_index]] = gm.predict_proba(train3[oof_val_index,:])[:,0]

            preds_gm[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits

    

    auc = roc_auc_score(train['target'], oof_bc_qda)

    print(f'AUC bc qda: {auc:.5}')



    auc = roc_auc_score(train['target'], oof_svc)

    print(f'AUC svc: {auc:.5}')



    auc = roc_auc_score(train['target'], oof_bc_knn)

    print(f'AUC bc knn: {auc:.5}')



    auc = roc_auc_score(train['target'], oof_gm)

    print(f'AUC gm: {auc:.5}')

    

    preds_blend1 = 0.68*preds_gm + 0.32*preds_svc

    preds_blend = 0.55*preds_bc_qda + 0.25*preds_blend1 + 0.2*preds_bc_knn



    oof_blend1 = 0.68*oof_gm + 0.32*oof_svc

    oof_blend = 0.55*oof_bc_qda + 0.25*oof_blend1 + 0.2*oof_bc_knn



    auc = roc_auc_score(train['target'], oof_blend)

    print(f'AUC blend: {auc:.7}')
import matplotlib.pyplot as plt

plt.hist(preds_blend,bins=100)

plt.title('Final Test.csv predictions')

plt.show()
sub = pd.read_csv('../input/sample_submission.csv')

if len(test) != 131073:

    sub['target'] = preds_blend

sub.to_csv('submission.csv',index=False)