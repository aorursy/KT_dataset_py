import numpy as np

import pandas as pd

from matplotlib import pylab as plt
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

train.shape
target = train['target']

train.drop('target', axis = 1, inplace = True)

target.value_counts()
test = pd.read_csv('../input/test.csv', index_col = 'id')

test.shape
index_train = train.index

index_test = test.index

print(len(index_train), len(index_test))
df_full = pd.concat([train, test], axis = 0)



del train, test

gc.collect()
df_stats = df_full.T.describe().T.drop('count', axis = 1)

df_stats.columns = ['source_' + c for c in df_stats.columns]

df_stats.head()
df_stats.shape
df_stats.loc[index_train].corrwith(target)
PARAMS = {}

PARAMS['random_state'] = 0

PARAMS['n_jobs'] = -1

PARAMS['C'] = .2

PARAMS['penalty'] = 'l1'

PARAMS['class_weight'] = 'balanced'

PARAMS['solver'] = 'saga'
logreg_scores = pd.DataFrame(columns = ['auc', 'acc', 'loss', 'tn', 'fn', 'fp', 'tp'])



def get_logreg_score(train_, target_):

    folds = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 20, random_state = 0)

    predict = pd.DataFrame(index = train_.index)

    

    # Cross-validation cycle

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(target_, target_)):

        train_x, train_y = train_.iloc[train_idx], target_.iloc[train_idx]

        valid_x, valid_y = train_.iloc[valid_idx], target_.iloc[valid_idx]

        

        clf = LogisticRegression(**PARAMS)

        clf.fit(train_x, train_y)

        predict[n_fold] = pd.Series(clf.predict_proba(valid_x)[:, 1], index = valid_x.index)



        del train_x, train_y, valid_x, valid_y

        gc.collect()

        

    predict = predict.mean(axis = 1)

    tn, fp, fn, tp = confusion_matrix(target_, (predict >= .5) * 1).ravel()

    return [

                 roc_auc_score(target_, predict), 

                 accuracy_score(target_, (predict >= .5) * 1), 

                 log_loss(target_, predict),

                 tn, fn, fp, tp

            ]
def get_submit(train_, test_, target_):

    predict = pd.DataFrame(index = test_.index)

    

    clf = LogisticRegression(**PARAMS)

    clf.fit(train_, target_)

    

    predict = pd.Series(clf.predict_proba(test_)[:, 1], index = test_.index).reset_index()

    predict.columns = ['id', 'target']

    

    return predict
step = 'source dataset'

logreg_scores = logreg_scores.T

logreg_scores[step] = get_logreg_score(df_full.loc[index_train], target)

logreg_scores = logreg_scores.T

logreg_scores
submit = get_submit(df_full.loc[index_train], df_full.loc[index_test], target)



score_auc = logreg_scores.loc[step, 'auc']

score_acc = logreg_scores.loc[step, 'acc']

score_loss = logreg_scores.loc[step, 'loss']

filename = 'subm_{}_{:.4f}_{:.4f}_{:.4f}_{}.csv'.format('source', score_auc, score_acc, score_loss,

                                                        datetime.now().strftime('%Y-%m-%d'))

print(filename)

submit.to_csv(filename, index = False)
dist_to_origin_sqr = (df_full**2).sum(axis = 1)

dist_to_origin_sqr.describe()
rad_sphere_sqr = 300

rad_sphere = np.sqrt(rad_sphere_sqr)

rad_sphere
df_stats['dist_to_sphere'] = np.sqrt(dist_to_origin_sqr) - rad_sphere

df_stats['dist_to_sphere'].describe()
np.corrcoef(df_stats['dist_to_sphere'].loc[index_train], target)[0, 1]
np.corrcoef(abs(df_stats['dist_to_sphere'].loc[index_train]), target)[0, 1]
mannwhitneyu(df_stats['dist_to_sphere'].loc[index_train], df_stats['dist_to_sphere'].loc[index_test])
df_full_sphere = (df_full * rad_sphere).divide(np.sqrt(dist_to_origin_sqr), axis = 'rows')

(df_full_sphere**2).sum(axis = 1).describe()
tmp = df_full_sphere.T.describe().T.drop('count', axis = 1)

tmp.columns = ['sphere_' + c for c in tmp.columns]

tmp.loc[index_train].corrwith(target)
df_stats = pd.concat([df_stats, tmp], axis = 1)



del tmp

gc.collect()



df_stats.head()
step = 'projection onto sphere'

logreg_scores = logreg_scores.T

logreg_scores[step] = get_logreg_score(df_full_sphere.loc[index_train], target)

logreg_scores = logreg_scores.T

logreg_scores
submit = get_submit(df_full_sphere.loc[index_train], df_full_sphere.loc[index_test], target)



score_auc = logreg_scores.loc[step, 'auc']

score_acc = logreg_scores.loc[step, 'acc']

score_loss = logreg_scores.loc[step, 'loss']

filename = 'subm_{}_{:.4f}_{:.4f}_{:.4f}_{}.csv'.format('full_sphere', score_auc, score_acc, score_loss,

                                                        datetime.now().strftime('%Y-%m-%d'))

print(filename)

submit.to_csv(filename, index = False)
df_signes = np.sign(df_full_sphere).astype(int)

df_signes.head()
df_signes.replace(-1, 2).astype(str).apply(lambda x: ''.join(x), axis = 1).nunique()
2**300
df_stats['positive_cnt'] = (df_signes > 0).sum(axis = 1)

df_stats['positive_cnt'].describe()
np.corrcoef(df_stats['positive_cnt'].loc[index_train], target)[0, 1]
mannwhitneyu(df_stats['positive_cnt'].loc[index_train], df_stats['positive_cnt'].loc[index_test])
step = 'quadrants'

logreg_scores = logreg_scores.T

logreg_scores[step] = get_logreg_score(df_signes.loc[index_train], target)

logreg_scores = logreg_scores.T

logreg_scores
submit = get_submit(df_signes.loc[index_train], df_signes.loc[index_test], target)



score_auc = logreg_scores.loc[step, 'auc']

score_acc = logreg_scores.loc[step, 'acc']

score_loss = logreg_scores.loc[step, 'loss']

filename = 'subm_{}_{:.4f}_{:.4f}_{:.4f}_{}.csv'.format('quad', score_auc, score_acc, score_loss,

                                                        datetime.now().strftime('%Y-%m-%d'))

print(filename)

submit.to_csv(filename, index = False)
df_stats['angle_w_bis'] = np.arccos(abs(df_full_sphere).sum(axis = 1) / rad_sphere_sqr)

df_stats['angle_w_bis'].describe()
np.corrcoef(df_stats['angle_w_bis'].loc[index_train], target)[0, 1]
mannwhitneyu(df_stats['angle_w_bis'].loc[index_train], df_stats['angle_w_bis'].loc[index_test])