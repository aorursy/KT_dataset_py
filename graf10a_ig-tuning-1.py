import time

import pickle

from tqdm import tqdm

from pathlib import Path

import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, QuantileTransformer

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture



from scipy.stats import rankdata

from sklearn.svm import SVC, NuSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
NFOLDS=5          # The number of folds

RS=42             # The random seed

debug=0          # The debuging mode switch: 1-debugging on; 0-debugging off 



params_2={'n_components' : 2,              # The parameters of the auxiliary 

          'init_params': 'random',         # GMM classifier computing clusters'

          'covariance_type': 'full',       # stats (means and covariance)

          'tol':0.001, 

          'reg_covar': 0.001, 

          'max_iter': 55, 

          'n_init': 10,

          'random_state': RS,

         }



params_2_qda={'n_components' : 2,              # The parameters of the auxiliary 

          'init_params': 'random',         # GMM classifier computing clusters'

          'covariance_type': 'full',       # stats (means and covariance)

          'tol':0.001, 

          'reg_covar': 0.001, 

          'max_iter': 50, 

          'n_init': 10,

          'random_state': RS,

         }



#PARAMETERS FOR THE GMM CLASSIFIER (2 clusters per class; 4 clusters in total)

params_4={'n_components' : 4, 

          'init_params': 'random', 

          'covariance_type': 'full', 

          'tol':0.001, 

          'reg_covar': 0.001, 

          'max_iter': 55, 

          'n_init': 10, 

          'random_state': RS,

         }



#PARAMETERS FOR THE QDA CLASSIFIER (2 clusters per class; 4 clusters in total)

params_qda={'reg_param' : 0.111,

         }



#PARAMETERS TO BE USED FOR PSEUDOLABELING GMM

low=0.0067

high=1-low



#PARAMETERS TO BE USED FOR PSEUDOLABELING QDA

# low_vals = [None, 0.01, 0.01, 0.01, 0.001, 0.0001]

# high_vals = [None, 0.99, 0.99, 0.99, 0.999, 0.9999]

# low_vals = [0.01, 0.01, 0.01, 0.001, 0.0001]

# high_vals = [0.99, 0.99, 0.99, 0.999, 0.9999]



rp_values = [0.8, 0.7, 0.6, 0.6]

#rp_values = [0.8]#[0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]



low_vals = [0.01, 0.01, 0.01, 0.01]

high_vals = [1-low for low in low_vals]



# low_vals.insert(0, None)

# high_vals.insert(0, None)



print(low_vals, high_vals)
%%time



path = Path('../input')



train = pd.read_csv(path/'train.csv')

test = pd.read_csv(path/'test.csv')

sub = pd.read_csv(path/'sample_submission.csv')
if debug:

    magic_max=2

    magic_min=0

    NFOLDS=2

else:

    magic_max=train['wheezy-copper-turtle-magic'].max()

    magic_min=train['wheezy-copper-turtle-magic'].min()
def preprocess(train=train, test=test):

       

    prepr = {} 

    

    #PREPROCESS 512 SEPARATE MODELS

    for i in range(magic_min, magic_max+1):



        # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS i     

        X = train[train['wheezy-copper-turtle-magic']==i].copy()

        Y = X.pop('target').values

        X_test = test[test['wheezy-copper-turtle-magic']==i].copy()

        idx_train = X.index 

        idx_test = X_test.index

        X.reset_index(drop=True,inplace=True)



        cols = np.array([c for c in X.columns if c not in ['id', 'wheezy-copper-turtle-magic']])



        l=len(X)

        X_all = pd.concat([X[cols], X_test[cols]], ignore_index=True)

        

        sel = VarianceThreshold(threshold=2)

        X_vt = sel.fit_transform(X_all)               # np.ndarray

        

        prepr['vt_' + str(i)] = X_vt

        prepr['n_vt' + str(i)] = X_vt.shape[1]

        prepr['feats_vt' + str(i)] = cols[sel.get_support(indices=True)]        

        prepr['train_size_' + str(i)] = l

        prepr['idx_train_' + str(i)] = idx_train

        prepr['idx_test_' + str(i)] = idx_test

        prepr['target_' + str(i)] = Y

        

    return prepr
%%time



data = preprocess()
def get_data(i, data):

    

    l = data['train_size_' + str(i)]

    

    X_all = data['vt_' + str(i)]                



    X = X_all[:l, :]

    X_test = X_all[l:, :]



    Y = data['target_' + str(i)]



    idx_train = data['idx_train_' + str(i)]

    idx_test = data['idx_test_' + str(i)]

    

    return X, X_test, Y, idx_train, idx_test
def initialize_cv():

    

    auc = np.array([])

    oof = np.zeros(len(train))

    preds = np.zeros(len(test)) 

    return preds, oof, auc
def report_results(oof, auc_all, clf_name='GMM'):

    

    # PRINT VALIDATION CV AUC FOR THE CLASSFIER

    print(f'The result summary for the {clf_name} classifier:')

    auc_combo = roc_auc_score(train['target'].values, oof)

    auc_av = np.mean(auc_all)

    std = np.std(auc_all)/(np.sqrt(NFOLDS)*np.sqrt(magic_max+1))



    print(f'The combined CV score is {round(auc_combo, 5)}.')    

    print(f'The folds average CV score is {round(auc_av, 5)}.')

    print(f'The standard deviation is {round(std, 5)}.\n')
def clusters_stats(X_train, Y_train, params=params_2):

    

    X_train_0 = X_train[Y_train==0]

    Y_train_0 = Y_train[Y_train==0].reshape(-1, 1)



    X_train_1 = X_train[Y_train==1]

    Y_train_1 = Y_train[Y_train==1].reshape(-1, 1)



    clf_0 = GaussianMixture(**params)



    clf_0.fit(X_train_0)

    means_0 = clf_0.means_

    covs_0 = clf_0.covariances_

    ps_0 = [np.linalg.inv(m) for m in covs_0]



    clf_1 = GaussianMixture(**params)



    clf_1.fit(X_train_1)

    means_1 = clf_1.means_

    covs_1 = clf_1.covariances_

    ps_1 = [np.linalg.inv(m) for m in covs_1]



    #SAVING CLUSTERS' MEANS AND COVARIANCES       

    ms = np.stack((means_0[0], means_0[1], means_1[0], means_1[1]))

    ps = np.stack((ps_0[0], ps_0[1], ps_1[0], ps_1[1]))

    

    return ms, ps
def pseudolabeling(X_train, X_test, Y_train, Y_pseudo, 

                   idx_test, test=test, low=low, high=high):

    

    assert len(test) == len(Y_pseudo), "The length of test does not match that of Y_pseudo!"

    

    #SELECT ONLY THE PSEUDOLABLES CORRESPONDING TO THE CURRENT VALUES OF 'wheezy-copper-turtle-magic'

    Y_aug = np.copy(Y_pseudo[idx_test])

    

    assert len(Y_aug) == len(X_test), "The length of Y_aug does not match that of X_test!"



    Y_aug[Y_aug > high] = 1

    Y_aug[Y_aug < low] = 0

    

    mask = (Y_aug == 1) | (Y_aug == 0)

    

    Y_useful = Y_aug[mask]

    X_test_useful = X_test[mask]

    

    X_train_aug = np.vstack((X_train, X_test_useful))

    Y_train_aug = np.vstack((Y_train.reshape(-1, 1), Y_useful.reshape(-1, 1)))

    

    return X_train_aug, Y_train_aug
def train_classifier(Y_pseudo, params=params_4):

    

    preds, oof, auc_all = initialize_cv()



    print(f"Computing centroids and covariances for the four clusters (two per class).")



    # BUILD 512 SEPARATE NON-LINEAR MODELS

    for i in tqdm(range(magic_min, magic_max+1)):   



        X, X_test, Y, idx_train, idx_test = get_data(i=i, data=data) 



        # STRATIFIED K FOLD

        auc_folds=np.array([])



        folds = StratifiedKFold(n_splits=NFOLDS, random_state=RS)



        for train_index, val_index in folds.split(X, Y):



            X_train, Y_train = X[train_index, :], Y[train_index]

            X_val, Y_val = X[val_index, :], Y[val_index]

            

            if Y_pseudo is None:

                params['means_init'], params['precisions_init'] = clusters_stats(X_train, Y_train)

            else:

                X_aug, Y_aug = pseudolabeling(X_train, X_test, Y_train, Y_pseudo, idx_test)

                params['means_init'], params['precisions_init'] = clusters_stats(X_aug, Y_aug.ravel())  

            

            #INSTANTIATING THE MAIN CLASSIFIER

            clf = GaussianMixture(**params) 

            

            clf.fit(np.concatenate([X_train, X_test], axis = 0))



            oof[idx_train[val_index]] = np.sum(clf.predict_proba(X_val)[:, 2:], axis=1)

            preds[idx_test] += np.sum(clf.predict_proba(X_test)[:,2: ], axis=1)/NFOLDS



            auc = roc_auc_score(Y_val, oof[idx_train[val_index]])

            auc_folds = np.append(auc_folds, auc)



        auc_all = np.append(auc_all, np.mean(auc_folds))



    report_results(oof, auc_all)

    

    return preds, oof, auc_all
Y_pseudo, oof, auc_all = train_classifier(Y_pseudo=None)
preds_gmm, oof_gmm, auc_gmm = train_classifier(Y_pseudo=Y_pseudo)



sub['target'] = preds_gmm

sub.to_csv('submission_gmm.csv',index=False)
def get_labels(X_train, Y_train, params=params_2_qda):

    

    X_train_0 = X_train[Y_train==0]

    X_train_1 = X_train[Y_train==1]



    clf_0 = GaussianMixture(**params)

    labels_0 = clf_0.fit_predict(X_train_0).reshape(-1, 1)



    clf_1 = GaussianMixture(**params)

    labels_1 = clf_1.fit_predict(X_train_1).reshape(-1, 1)

    

    labels_1[labels_1==0] = 2

    labels_1[labels_1==1] = 3



    #CREATE LABELED DATA 

    

    X_l = np.vstack((X_train_0, X_train_1))

    Y_l = np.vstack((labels_0, labels_1))

    

    perm = np.random.permutation(len(X_l))

    

    X_l = X_l[perm]

    Y_l = Y_l[perm]

    

    return X_l, Y_l
def pseudolabeling_qda(X_train, X_test, Y_train, Y_pseudo, 

                       idx_test, low, high, test=test):

    

    assert len(test) == len(Y_pseudo), "The length of test does not match that of Y_pseudo!"

    

    #SELECT ONLY THE PSEUDOLABLES CORRESPONDING TO THE CURRENT VALUES OF 'wheezy-copper-turtle-magic'

    Y_aug = np.copy(Y_pseudo[idx_test])

    

    assert len(Y_aug) == len(X_test), "The length of Y_aug does not match that of X_test!"



    Y_aug[Y_aug > high] = 1

    Y_aug[Y_aug < low] = 0

    

    mask = (Y_aug == 1) | (Y_aug == 0)

    

    Y_useful = Y_aug[mask]

    X_test_useful = X_test[mask]

    

    X_train_aug = np.vstack((X_train, X_test_useful))

    Y_train_aug = np.vstack((Y_train.reshape(-1, 1), Y_useful.reshape(-1, 1)))

    

    return X_train_aug, Y_train_aug
def train_qda(Y_pseudo, low, high, params=params_qda):

    

    preds, oof, auc_all = initialize_cv()



    print(f"Computing centroids and covariances for the four clusters (two per class).")



    # BUILD 512 SEPARATE NON-LINEAR MODELS

    for i in tqdm(range(magic_min, magic_max+1)):   



        X, X_test, Y, idx_train, idx_test = get_data(i=i, data=data) 



        # STRATIFIED K FOLD

        auc_folds=np.array([])



        folds = StratifiedKFold(n_splits=NFOLDS, random_state=RS)



        for train_index, val_index in folds.split(X, Y):



            X_train, Y_train = X[train_index, :], Y[train_index]

            X_val, Y_val = X[val_index, :], Y[val_index]

            

            #INSTANTIATING THE MAIN CLASSIFIER

            clf = QuadraticDiscriminantAnalysis(**params)  

            

            if Y_pseudo is None:

                X_l, Y_l = get_labels(X_train, Y_train)

            else:

                X_aug, Y_aug = pseudolabeling_qda(X_train, X_test, Y_train, Y_pseudo, idx_test, low, high)

                X_l, Y_l = get_labels(X_aug, Y_aug.ravel()) 

                

            clf.fit(X_l, Y_l.ravel())

                

            oof[idx_train[val_index]] = np.sum(clf.predict_proba(X_val)[:, 2:], axis=1)

            preds[idx_test] += np.sum(clf.predict_proba(X_test)[:,2: ], axis=1)/NFOLDS



            auc = roc_auc_score(Y_val, oof[idx_train[val_index]])

            auc_folds = np.append(auc_folds, auc)



        auc_all = np.append(auc_all, np.mean(auc_folds))



    report_results(oof, auc_all, clf_name='QDA')

    

    return preds, oof, auc_all
Y_pseudo=preds_gmm#None



for rp, low, high in zip(rp_values, low_vals, high_vals):

    parmas_qda = {'reg_param': rp}

    Y_pseudo, oof_qda, auc_qda = train_qda(Y_pseudo=Y_pseudo, low=low, high=high, params=params_qda)



#THE LAST PREDICTIONS 

preds_qda = Y_pseudo



sub['target'] = preds_qda

sub.to_csv('submission_qda.csv',index=False)
preds_highest = preds_gmm

oof_highest = oof_gmm



mask = (auc_qda > auc_gmm)



print(f"The number of models where QDA's predictions are better is {sum(mask)}.")
for i in tqdm(range(magic_min, magic_max+1)):

    

    if mask[i]:

        _, _, _, idx_train, idx_test = get_data(i=i, data=data)

        oof_highest[idx_train] = oof_qda[idx_train]

        preds_highest[idx_test] = preds_qda[idx_test]

        

auc = roc_auc_score(train['target'].values, oof_highest)

print(f"The 'highest' ROC AUC score is {auc}.")
sub['target'] = preds_highest

sub.to_csv('submission_highest.csv',index=False)
oof_all = pd.DataFrame()

preds_all = pd.DataFrame()



oof_all['gmm'] = rankdata(oof_gmm)/len(oof_gmm)

oof_all['qda'] = rankdata(oof_qda)/len(oof_qda)

preds_all['gmm'] = rankdata(preds_gmm)/len(preds_gmm)

preds_all['qda'] = rankdata(preds_qda)/len(preds_qda)



lr = LogisticRegression()



lr.fit(oof_all.values, train['target'].values)

preds_lr = lr.predict_proba(preds_all.values)[:,1]



preds_train = lr.predict_proba(oof_all)[:,1]



auc = roc_auc_score(train['target'].values, preds_train)

print(f"The final ROC AUC score is {auc}.")
sub['target'] = preds_lr

sub.to_csv('submission_lr.csv',index=False)
w = 0.02

mask = (preds_gmm < (0.5 + w))&(preds_gmm > (0.5 - w))



preds = rankdata(preds_gmm)/len(preds_gmm)



preds[mask] = preds_lr[mask]



sub['target'] = preds

sub.to_csv('submission_picking.csv',index=False)