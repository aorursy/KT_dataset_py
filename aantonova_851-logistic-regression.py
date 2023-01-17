ver = 'logreg'
import numpy as np

import pandas as pd
import gc

import time

from datetime import datetime

import warnings

warnings.simplefilter(action = 'ignore')
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
train = pd.read_csv('../input/train.csv', index_col = 'id')

train.head()
target = train['target']

train.drop('target', axis = 1, inplace = True)

target.value_counts()
test = pd.read_csv('../input/test.csv', index_col = 'id')

test.head()
def save_submit(test_, clfs_, filename):

    subm = pd.DataFrame(np.zeros(test_.shape[0]), index = test_.index, columns = ['target'])

    for clf in clfs_:

        subm['target'] += clf.predict_proba(test_)[:, 1]

    subm['target'] /= len(clfs_)

    subm = subm.reset_index()

    subm.columns = ['id', 'target']

    subm.to_csv(filename, index = False)
scores = pd.DataFrame(columns = ['auc', 'acc', 'loss', 'tn', 'fn', 'fp', 'tp'])
mw = pd.DataFrame(index = ['stat', 'p'])

for c in train.columns:

    mw[c] = mannwhitneyu(train[c], test[c])

mw = mw.T

mw[mw['p'] < .01].shape
bad_features = list(mw[mw['p'] < .01].index)

len(bad_features)
def logreg_cross_validation(train_, target_, params,

                            num_folds = 5, repeats = 20, rs = 0):

    

    print(params)

    

    clfs = []

    folds = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = repeats, random_state = rs)

    

    valid_pred = pd.DataFrame(index = train_.index)

    

    # Cross-validation cycle

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(target_, target_)):

        print('--- Fold {} started at {}'.format(n_fold, time.ctime()))

        

        train_x, train_y = train_.iloc[train_idx], target_.iloc[train_idx]

        valid_x, valid_y = train_.iloc[valid_idx], target_.iloc[valid_idx]

        

        clf = LogisticRegression(**params)

        clf.fit(train_x, train_y)

    

        clfs.append(clf)



        predict = clf.predict_proba(valid_x)[:, 1]

    

        tn, fp, fn, tp = confusion_matrix(valid_y, (predict >= .5) * 1).ravel()

        auc = roc_auc_score(valid_y, predict)

        acc = accuracy_score(valid_y, (predict >= .5) * 1)

        loss = log_loss(valid_y, predict)

        print('TN =', tn, 'FN =', fn, 'FP =', fp, 'TP =', tp)

        print('AUC = ', auc, 'Loss =', loss, 'Acc =', acc)

        

        valid_pred[n_fold] = pd.Series(predict, index = valid_x.index)



        del train_x, train_y, valid_x, valid_y, predict

        gc.collect()



    return clfs, valid_pred
params = {}

params['random_state'] = 0

params['n_jobs'] = -1

params['C'] = .2

params['penalty'] = 'l1'

params['class_weight'] = 'balanced'

params['solver'] = 'saga'
clfs, pred = logreg_cross_validation(train.drop(bad_features, axis = 1), target, params)
pred_mean = pred.mean(axis = 1)
scores = scores.T

tn, fp, fn, tp = confusion_matrix(target, (pred_mean >= .5) * 1).ravel()

scores['logreg'] = [

                 roc_auc_score(target, pred_mean), 

                 accuracy_score(target, (pred_mean >= .5) * 1), 

                 log_loss(target, pred_mean),

                 tn, fn, fp, tp]



scores = scores.T

scores
score_auc = scores.loc['logreg', 'auc']

score_acc = scores.loc['logreg', 'acc']

score_loss = scores.loc['logreg', 'loss']

print(score_auc, score_acc, score_loss)
loc_ver = 'v1'

filename = 'subm_{}_{}_{:.4f}_{:.4f}_{:.4f}_{}.csv'.format(ver, loc_ver, score_auc, score_acc, score_loss,

                                                        datetime.now().strftime('%Y-%m-%d'))

print(filename)

save_submit(test.drop(bad_features, axis = 1), clfs, filename)