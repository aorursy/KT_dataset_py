import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime as dt

import time 



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import log_loss, auc, roc_curve

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import lightgbm as lgb
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.shape
train.head(2)
# set random seed to use

RS = 42



def build_clf(clf, params):

    if clf == 'logistic':

        return LogisticRegression(**params)

    elif clf == 'sgd':

        return SGDClassifier(**params)

    elif clf == 'gnb':

        return GaussianNB(**params)

    elif clf == 'knn':

        return KNeighborsClassifier(**params)

    elif clf == 'rf':

        return RandomForestClassifier(**params)

    elif clf == 'lgb':

        return lgb.LGBMClassifier(**params)

    elif clf == 'svm':

        return SVC(**params)

    else:

        raise ValueError('{} is not valid, choose valid classifier.'.format(clf))





def train_cv(clf_choice, params, xtrain, ytrain, n_folds=10, fit_params=None):

    print('Training with: {}'.format(clf_choice))

    clfs = []

    lls = []

    aucs = []



    folds = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.15, random_state=RS)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(ytrain, ytrain)):

        x_trn, y_trn = xtrain.iloc[trn_idx].copy(), ytrain[trn_idx]

        x_val, y_val = xtrain.iloc[val_idx].copy(), ytrain[val_idx]

        

        # train

        clf = build_clf(clf_choice, params)

        if fit_params is None:

            clf.fit(x_trn, y_trn)

        else:

            # just for lgb

            fit_params['eval_set'] = (x_val, y_val)

            clf.fit(x_trn, y_trn, **fit_params)

        # pred train

        trn_pred = clf.predict_proba(x_trn)[:, 1]

        # compute auc

        fpr, tpr, thresholds = roc_curve(y_trn, trn_pred, pos_label=1)

        trn_auc_i = auc(fpr, tpr)

        # logloss

        trn_logloss_i = log_loss(y_trn, trn_pred)

        

        # pred val

        val_pred = clf.predict_proba(x_val)[:, 1]

        # compute auc

        fpr, tpr, thresholds = roc_curve(y_val, val_pred, pos_label=1)

        val_auc_i = auc(fpr, tpr)

        # logloss

        val_logloss_i = log_loss(y_val, val_pred)

        print('[cv{0}] train: auc={1:.4f}, logloss={2:.4f} | val: auc={3:.4f}, logloss={4:.4f}'

              .format(fold_, trn_auc_i, trn_logloss_i, val_auc_i, val_logloss_i))

        aucs.append(val_auc_i)

        

        clfs.append(clf)

    mauc = np.mean(aucs)

    print('Done cv, mean auc: {0:.4f}'.format(mauc))

    return clfs, mauc





def test_sub(clfs, clf_choice, xtest, cvscore, use_cols=None):

    test_ids = xtest['id']

    if use_cols is not None:

        print('Number of use_cols: {}'.format(len(use_cols)))

        cols = use_cols

    else:

        cols = [c for c in xtest.columns if c != 'id']



    t1 = time.time()

    filename = '{0}_sub_{1:.6f}_{2}.csv'.format(clf_choice, cvscore, dt.now().strftime('%m-%d-%H-%M'))

    print('Saving test sub to {}'.format(filename))



    mfull = np.zeros(len(xtest))

    for n, clf in enumerate(clfs):

        print('generating sub with fold {}'.format(n))

        mfull += clf.predict_proba(xtest[cols])[:, 1] / len(clfs)

    # save

    xsub = pd.DataFrame()

    xsub['id'] = test_ids

    xsub['target'] = mfull

    xsub.to_csv(filename, index=False)

    t2 = time.time()

    print('Done saving {0:} took {1:.2f}mins'.format(filename, (t2-t1)/60))
# get target and drop ids 

ytrain = train.target.values

xtrain = train.drop(['id', 'target'], axis=1)
clf_choice = 'logistic'

logistic_params = {'solver': 'liblinear', 'class_weight':'balanced', 'penalty': 'l1', 'C': 0.1, 'random_state': RS}

log_clfs, log_mauc = train_cv(clf_choice, logistic_params, xtrain, ytrain)



# sub

# test_sub(log_clfs, clf_choice, test, log_mauc)
svm_params = {'C': 1e-3, 'kernel': 'rbf', 'probability':True, 'gamma':'auto', 'random_state':RS}

svm_clfs, svm_mauc = train_cv('svm', svm_params, xtrain, ytrain)

# sub

# test_sub(svm_clfs, clf_choice, test, svm_mauc, use_cols)
# feature importance

selector = np.ones(xtrain.shape[1]).astype(bool)

for clf in log_clfs:

    selector = (selector) & (clf.coef_[0] == 0)

print('total number of zero importance features: {}'.format(sum(selector)))

use_cols = xtrain.columns[~selector]

print(xtrain.shape, xtrain[use_cols].shape)

clfs, mauc = train_cv('logistic', logistic_params, xtrain[use_cols], ytrain)
use_cols
clf_choice = 'svm'

RS=42

svm_params = {'C': 1e-3, 'kernel': 'rbf', 'probability':True, 'gamma':'auto', 'random_state':RS}

svm_clfs, svm_mauc = train_cv(clf_choice, svm_params, xtrain[use_cols], ytrain)



# sub

test_sub(svm_clfs, clf_choice, test, svm_mauc, use_cols)
# similarly you can also try to run GNB with/wihtout the selected features