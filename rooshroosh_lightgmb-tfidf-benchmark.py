# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import time
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
train = pd.read_csv('../input/train.csv')
train.groupby('target').count()
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(min_df=3, stop_words='english', ngram_range=(1,2))
X = vec.fit_transform(train['question_text'])
Y = train['target'].values
spliter = StratifiedKFold(n_splits=5,shuffle=True, random_state=1)
FOLD_LIST = list(spliter.split(Y,Y))
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 15,
    'num_leaves': 64,  # 63, 127, 255
    'feature_fraction': 0.05, # 0.1, 0.01
    'bagging_fraction': 0.8,
    'verbose': 1
}
# dim_learning_rate = Real(low=1e-3, high=0.99, prior='log-uniform',name='learning_rate')
# dim_estimators = Integer(low=50, high=6000,name='n_estimators')
# dim_max_depth = Integer(low=1, high=15,name='max_depth')

# dimensions = [dim_learning_rate,
#               dim_estimators,
#               dim_max_depth]

# default_parameters = [0.3,100,3]
# def createModel(learning_rate,n_estimators,max_depth):       

#     oof_preds = np.zeros(shape=(Y.shape[0], 2))
#     for fold_, (trn_, val_) in enumerate(FOLD_LIST):
#         trn_x, trn_y = X[trn_], Y[trn_]
#         val_x, val_y = X[val_], Y[val_]

#         clf = lgb.LGBMClassifier(**lgb_params,learning_rate=learning_rate,
#                                 n_estimators=n_estimators,max_depth=max_depth)
#         clf.fit(
#             trn_x, trn_y,
#             eval_set=[(trn_x, trn_y), (val_x, val_y)],
#             verbose=False,
#             early_stopping_rounds=50
#         )
#         oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
#         print('fold',fold_+1, 
#               roc_auc_score(
#                   val_y,
#                   np.where(clf.predict_proba(val_x, num_iteration=clf.best_iteration_)>0.5, 1,0)[:,0]
#               )
#         )
# #         clfs.append(clf)
#     loss = roc_auc_score(y_true=Y,  y_score=oof_preds[:,0])    
#     return loss
# iter_count = 0
# @use_named_args(dimensions=dimensions)
# def fitness(learning_rate,n_estimators,max_depth):
#     """
#     Hyper-parameters:
#     learning_rate:     Learning-rate for the optimizer.
#     n_estimators:      Number of estimators.
#     max_depth:         Maximum Depth of tree.
#     """
#     global iter_count
#     iter_count+=1
#     # Print the hyper-parameters.
#     print('iteration:', iter_count)
#     print('learning rate: {0:.2e}'.format(learning_rate), 'estimators:', n_estimators, 'max depth:', max_depth)
    
#     lv= createModel(learning_rate=learning_rate,
#                     n_estimators=n_estimators,
#                     max_depth = max_depth)
#     return lv
# error = fitness(default_parameters)
# # use only if you haven't found out the optimal parameters for xgb. else comment this block.
# search_result = gp_minimize(func=fitness,
#                             dimensions=dimensions,
#                             acq_func='EI', # Expected Improvement.
#                             n_calls=70,
#                            x0=default_parameters)
# plot_convergence(search_result)
# plt.show()
# print(search_result.x)
# learning_rate = search_result.x[0]
# n_estimators = search_result.x[1]
# max_depth = search_result.x[2]
test = pd.read_csv('../input/test.csv')
X_target = vec.transform(test.question_text)
Y_target = []
off_predict = np.zeros(shape=Y.shape)
for fold_id,(train_idx, val_idx) in enumerate(FOLD_LIST):
    print('FOLD:',fold_id)
    X_train = X[train_idx]
    y_train = Y[train_idx]
    X_valid = X[val_idx]
    y_valid = Y[val_idx]
    
    lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=vec.get_feature_names(),
    #             categorical_feature = categorical
                         )

    lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=vec.get_feature_names(),
    #             categorical_feature = categorical
                         )

    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=36000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=200,
        verbose_eval=1000
    )
    off_predict[val_idx] = lgb_clf.predict(X_valid)
    test_pred = lgb_clf.predict(X_target)
    Y_target.append(np.log1p(test_pred))
    print('fold finish after', time.time()-modelstart)
off_predict.shape
Y_target = np.array(Y_target)
from sklearn import  metrics

_thresh = []
for thresh in np.arange(0.1, 0.501, 0.01):
    _thresh.append([thresh, metrics.f1_score(Y, (off_predict>thresh).astype(int))])
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(Y, (off_predict>thresh).astype(int))))
_thresh = np.array(_thresh)
best_id = _thresh[:,1].argmax()
best_thresh = _thresh[best_id][0]
best_thresh
# 0.11, 0.13, 1000)
_thresh = []
for thresh in np.linspace(best_thresh-0.05, best_thresh+0.05, 100):
    _thresh.append([thresh, metrics.f1_score(Y, (off_predict>thresh).astype(int))])
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(Y, (off_predict>thresh).astype(int))))
_thresh = np.array(_thresh)
best_id = _thresh[:,1].argmax()
best_thresh = _thresh[best_id][0]
!head ./../input/sample_submission.csv
sub = pd.read_csv('../input/sample_submission.csv')
sub['prediction'] = np.where(np.expm1(Y_target.mean(axis=0))>best_thresh, 1,0,)
sub.to_csv('submission.csv', index=False)
