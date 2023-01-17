# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostClassifier

from bayes_opt import BayesianOptimization



import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
raws_features = train_data.columns[2:]

train_X, train_y = train_data[raws_features], train_data['target']

test_X = test_data[raws_features]
n_splits = 10

num_round = 77777

seed = 7777
class CVClassifier():

    def __init__(self, estimator, n_splits=5, stratified=True, num_round=77777, **params):

        self.n_splits_ = n_splits

        self.scores_ = []

        self.clf_list_ = []

        self.estimator_ = estimator

        self.stratified_ = stratified

        self.num_round_ = num_round

        if params:

            self.params_ = params

        

    def cv(self, train_X, train_y):

        if self.stratified_:

            folds = StratifiedKFold(self.n_splits_, shuffle=True, random_state=seed)

        else:

            folds = KFold(self.n_splits_, shuffle=True, random_state=seed)

        oof = np.zeros(len(train_y))

        for fold, (train_idx, val_idx) in enumerate(folds.split(train_X, train_y)):

            print('fold %d' % fold)

            trn_data, trn_y = train_X.iloc[train_idx], train_y[train_idx]

            val_data, val_y = train_X.iloc[val_idx], train_y[val_idx]

            if self.estimator_ == 'lgbm':

                train_set = lgb.Dataset(data=trn_data, label=trn_y)

                val_set = lgb.Dataset(data=val_data, label=val_y)

                clf = lgb.train(params=params, train_set=train_set, num_boost_round=num_round, 

                                valid_sets=[train_set, val_set], verbose_eval=100, early_stopping_rounds=200)

                oof[val_idx] = clf.predict(train_X.iloc[val_idx], num_iteration=clf.best_iteration)

                

            elif self.estimator_ == 'xgb':

                train_set = xgb.DMatrix(data=trn_data, label=trn_y)

                val_set = xgb.DMatrix(data=val_data, label=val_y)

                watchlist = [(train_set, 'train'), (val_set, 'valid')]

                clf = xgb.train(self.params_, train_set, self.num_round_, watchlist, 

                               early_stopping_rounds=200, verbose_eval=100)

                oof[val_idx] = clf.predict(val_set, ntree_limit=clf.best_ntree_limit)

            

            elif self.estimator_ == 'cat':

                clf = CatBoostClassifier(self.num_round_, task_type='GPU', early_stopping_rounds=500, **self.params_)

                clf.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=500)

                oof[val_idx] = clf.predict_proba(val_data)[:, 1]



            # sk-learn model

            else:

                clf = self.estimator_.fit(trn_data, trn_y)

                try:

                    oof[val_idx] = clf.predict_proba(val_data)[:, 1]

                except AttributeError:

                    oof[val_idx] = clf.decision_function(val_data)

            

            self.clf_list_.append(clf)

            fold_score = roc_auc_score(train_y[val_idx], oof[val_idx])

            self.scores_.append(fold_score)

            print('Fold score: {:<8.5f}'.format(fold_score))

        self.oof_ = oof

        self.score_ = roc_auc_score(train_y, oof)

        print("CV score: {:<8.5f}".format(self.score_))

        

    def predict(self, test_X):

        self.predictions_ = np.zeros(len(test_X))

        

        if self.estimator_ == 'lgbm':

            self.feature_importance_df_ = pd.DataFrame()

            for fold, clf in enumerate(self.clf_list_):

                fold_importance_df = pd.DataFrame()

                fold_importance_df["feature"] = features

                fold_importance_df["importance"] = clf.feature_importance()

                fold_importance_df["fold"] = fold + 1

                self.feature_importance_df_ = pd.concat([self.feature_importance_df_, fold_importance_df], axis=0)

                

                self.predictions_ += clf.predict(test_X, num_iteration=clf.best_iteration) * (self.scores_[fold] / sum(self.scores_))

        elif self.estimator_ == 'xgb':

            for fold, clf in enumerate(self.clf_list_):

                self.predictions_ += clf.predict(xgb.DMatrix(test_X), ntree_limit=clf.best_ntree_limit) \

                * (self.scores_[fold] / sum(self.scores_))

        elif self.estimator_ == 'cat':

            for fold, clf in enumerate(self.clf_list_):

                self.predictions_ += clf.predict_proba(test_X)[:, 1] * (self.scores_[fold] / sum(self.scores_))

        else:

            for fold, clf in enumerate(self.clf_list_):

                self.predictions_ += clf.predict_proba(test_X)[:, 1] * (self.scores_[fold] / sum(self.scores_))
# Class for Bayesian Optimisation

class CVForBO():

    def __init__(self, model, train_X, train_y, test_X, base_params, int_params=[], n_splits=5, num_round=77777):

        self.oofs_ = []

        self.params_ = []

        self.predictions_ = []

        self.cv_scores_ = []

        self.model_ = model

        self.train_X_ = train_X

        self.train_y_ = train_y

        self.test_X_ = test_X

        self.base_params_ = base_params

        self.int_params_ = int_params

        self.n_splits_ = n_splits

        self.num_round_ = num_round

        

    def cv(self, **opt_params):

        for p in self.int_params_:

            if p in opt_params:

                opt_params[p] = int(np.round(opt_params[p]))

        self.base_params_.update(opt_params)

        

        cv_model = CVClassifier(self.model_, n_splits=self.n_splits_, num_round=self.num_round_, **self.base_params_)

        cv_model.cv(self.train_X_, self.train_y_)

        cv_model.predict(self.test_X_)

        

        self.oofs_.append(cv_model.oof_)

        self.predictions_.append(cv_model.predictions_)

        self.params_.append(self.base_params_)

        self.cv_scores_.append(cv_model.score_)



        return cv_model.score_

    

    def post_process(self, model_type=None, oof_path='inter_oofs.csv', pred_path='inter_preds.csv', params_path='inter_params.csv'):

        if not model_type:

            model_type=self.model_

        cols = ['{}_{}_{}'.format(model_type, str(self.cv_scores_[k]).split('.')[-1][:5], k) for k in range(len(self.cv_scores_))]

        self.oof_df = pd.DataFrame(np.array(self.oofs_).T, columns=cols)

        self.pred_df = pd.DataFrame(np.array(self.predictions_).T, columns=cols)

        self.params_df = pd.DataFrame(self.params_).T.rename(columns={c_old: c_new for c_old, c_new in enumerate(cols)})

        

        self.oof_df.to_csv(oof_path)

        self.pred_df.to_csv(pred_path)

        self.params_df.to_csv(params_path)
cat_params = {

    'eval_metric': 'AUC',

    'bootstrap_type': 'Bernoulli',

    'objective': 'Logloss',

    'od_type': 'Iter',

    'random_seed': seed,

    'allow_writing_files': False}



cv_cat_for_BO = CVForBO('cat', train_X, train_y, test_X, cat_params, ['depth'])

cat_BO = BayesianOptimization(cv_cat_for_BO.cv, {

    'depth': (2, 4), 

    'l2_leaf_reg': (37, 97), 

    'random_strength': (5, 17), 

    'eta': (0.01, 0.1)

    }, random_state=seed)



cat_BO.maximize(init_points=2, n_iter=15, acq='ei')
print(cat_BO.max)

cv_cat_for_BO.post_process()
max_idx = cv_cat_for_BO.cv_scores_.index(cat_BO.max['target'])



sub_df = pd.DataFrame({'ID_code': test_data['ID_code'], 

                      'target': cv_cat_for_BO.predictions_[max_idx]})

sub_df.to_csv('submissions.csv', index=False)