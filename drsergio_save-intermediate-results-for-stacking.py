import gc

import os

import numpy as np

import pandas as pd

import xgboost as xgb

import lightgbm as lgb

import matplotlib.pyplot as plt



from time import time

from scipy.stats import norm, rankdata

from catboost import CatBoostClassifier

from bayes_opt import BayesianOptimization

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score, auc

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, MinMaxScaler, RobustScaler



plt.style.use('bmh')

plt.rcParams['figure.figsize'] = (10, 10)



print(os.listdir("../input"))
seed = 1337

precision = 5



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
index_column = 'ID_code'

target_column = 'target'



raws_features = train_data.columns[2:]

train_X, train_y = train_data[raws_features], train_data[target_column]

test_X = test_data[raws_features]
class CVClassifier():

    def __init__(self, estimator, n_splits=5, stratified=True, num_round=20000, early_stop=200, shuffle=False, **params):

        self.n_splits_ = n_splits

        self.scores_ = []

        self.clf_list_ = []

        self.estimator_ = estimator

        self.stratified_ = stratified

        self.num_round_ = num_round

        self.early_stop = early_stop

        self.shuffle = shuffle

        if params:

            self.params_ = params

        

    def cv(self, train_X, train_y):

        if self.stratified_:

            folds = StratifiedKFold(self.n_splits_, shuffle=self.shuffle, random_state=seed)

        else:

            folds = KFold(self.n_splits_, shuffle=self.shuffle, random_state=seed)

        oof = np.zeros(len(train_y))

        for fold, (train_idx, val_idx) in enumerate(folds.split(train_X, train_y)):

            print('fold %d' % (fold + 1))

            trn_data, trn_y = train_X.iloc[train_idx], train_y[train_idx]

            val_data, val_y = train_X.iloc[val_idx], train_y[val_idx]

            if self.estimator_ == 'lgbm':

                train_set = lgb.Dataset(data=trn_data.values, label=trn_y.values)

                val_set = lgb.Dataset(data=val_data.values, label=val_y.values)              

                clf = lgb.train(params=params, train_set=train_set, num_boost_round=self.num_round_, valid_sets=[train_set, val_set], 

                                verbose_eval=500, early_stopping_rounds=self.early_stop)

                oof[val_idx] = clf.predict(train_X.iloc[val_idx], num_iteration=clf.best_iteration)

                

            elif self.estimator_ == 'xgb':

                train_set = xgb.DMatrix(data=trn_data, label=trn_y)

                val_set = xgb.DMatrix(data=val_data, label=val_y)

                watchlist = [(train_set, 'train'), (val_set, 'valid')]

                clf = xgb.train(self.params_, train_set, self.num_round_, watchlist, early_stopping_rounds=self.early_stop, verbose_eval=500)

                oof[val_idx] = clf.predict(val_set, ntree_limit=clf.best_ntree_limit)

            

            elif self.estimator_ == 'cat':

                clf = CatBoostClassifier(self.num_round_, task_type='GPU', early_stopping_rounds=self.early_stop, **self.params_)

                # clf = CatBoostClassifier(self.num_round_, early_stopping_rounds=self.early_stop, **self.params_)

                clf.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=500)

                oof[val_idx] = clf.predict_proba(val_data)[:, 1]



            # sklearn model

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

        

        print('\nKFold CV: %s' % list(map(lambda x: np.round(x, precision), list(self.scores_))))

        print('Mean Kfold CV: {0} +/- {1}'.format(np.round(np.mean(self.scores_), precision), np.round(np.std(list(self.scores_)), precision)))

        print("Estimated CV: {0:0.5f}".format(self.score_))

        

    def predict(self, test_X):

        self.predictions_ = np.zeros(len(test_X))

        

        if self.estimator_ == 'lgbm':

            self.feature_importance_df_ = pd.DataFrame()

            for fold, clf in enumerate(self.clf_list_):

                fold_importance_df = pd.DataFrame()

                fold_importance_df["feature"] = clf.feature_name()

                fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

                fold_importance_df["fold"] = fold + 1

                self.feature_importance_df_ = pd.concat([self.feature_importance_df_, fold_importance_df], axis=0)

                # self.predictions_ += clf.predict(test_X, num_iteration=clf.best_iteration) * (self.scores_[fold] / sum(self.scores_))

                self.predictions_ += clf.predict(test_X, num_iteration=clf.best_iteration) / self.n_splits_     

        elif self.estimator_ == 'xgb':

            for fold, clf in enumerate(self.clf_list_):

                # self.predictions_ += clf.predict(xgb.DMatrix(test_X), ntree_limit=clf.best_ntree_limit) * (self.scores_[fold] / sum(self.scores_))

                self.predictions_ += clf.predict(xgb.DMatrix(test_X), ntree_limit=clf.best_ntree_limit) / self.n_splits_

        elif self.estimator_ == 'cat':

            for fold, clf in enumerate(self.clf_list_):

                # self.predictions_ += clf.predict_proba(test_X)[:, 1] * (self.scores_[fold] / sum(self.scores_))

                self.predictions_ += clf.predict_proba(test_X)[:, 1] / self.n_splits_

        else:

            for fold, clf in enumerate(self.clf_list_):

                # self.predictions_ += clf.predict_proba(test_X)[:, 1] * (self.scores_[fold] / sum(self.scores_))

                self.predictions_ += clf.predict_proba(test_X)[:, 1] / self.n_splits_
# Class for Bayesian Optimisation

class CVForBO():

    def __init__(self, model, train_X, train_y, test_X, base_params, int_params=[], n_splits=5, num_round=20000, early_stop=200, shuffle=False):

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

        self.early_stop = early_stop

        self.shuffle = shuffle

        

    def cv(self, **opt_params):

        for p in self.int_params_:

            if p in opt_params:

                opt_params[p] = int(np.round(opt_params[p]))

        self.base_params_.update(opt_params)

        

        cv_model = CVClassifier(self.model_, n_splits=self.n_splits_, num_round=self.num_round_, early_stop=self.early_stop, shuffle=self.shuffle, **self.base_params_)

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
def standard_scaler(train_X, test_X):

    train_X_df = train_X.copy()

    test_X_df = test_X.copy()

    scaler = StandardScaler()

    train_X_df = pd.DataFrame(scaler.fit_transform(train_X_df), columns=train_X_df.columns)

    test_X_df = pd.DataFrame(scaler.transform(test_X_df), columns=test_X_df.columns)

    return train_X_df, test_X_df



def quantile_transformer(train_X, test_X):

    train_X_df = train_X.copy()

    test_X_df = test_X.copy()

    transformer = QuantileTransformer(output_distribution='normal', random_state=seed)

    train_X_df = pd.DataFrame(transformer.fit_transform(train_X_df), columns=train_X_df.columns)

    test_X_df = pd.DataFrame(transformer.transform(test_X_df), columns=test_X_df.columns)

    return train_X_df, test_X_df



def min_max_scaler(train_X, test_X):

    train_X_df = train_X.copy()

    test_X_df = test_X.copy()

    scaler = MinMaxScaler()

    train_X_df = pd.DataFrame(scaler.fit_transform(train_X_df), columns=train_X_df.columns)

    test_X_df = pd.DataFrame(scaler.transform(test_X_df), columns=test_X_df.columns)

    return train_X_df, test_X_df



def power_transformer(train_X, test_X):

    train_X_df = train_X.copy()

    test_X_df = test_X.copy()

    scaler = PowerTransformer()

    train_X_df = pd.DataFrame(scaler.fit_transform(train_X_df), columns=train_X_df.columns)

    test_X_df = pd.DataFrame(scaler.transform(test_X_df), columns=test_X_df.columns)

    return train_X_df, test_X_df



def robust_scaler(train_X, test_X):

    train_X_df = train_X.copy()

    test_X_df = test_X.copy()

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))

    train_X_df = pd.DataFrame(scaler.fit_transform(train_X_df), columns=train_X_df.columns)

    test_X_df = pd.DataFrame(scaler.transform(test_X_df), columns=test_X_df.columns)

    return train_X_df, test_X_df



def features_generation(train_X, test_X):

    train_X_df = train_X.copy()

    test_X_df = test_X.copy()

    train_test_df = train_X_df.append(test_X_df)

    

    # Normalize the data, so that it can be used in norm.cdf(), as though it is a standard normal variable

    scaler = StandardScaler()

    train_test_df = pd.DataFrame(scaler.fit_transform(train_test_df), columns=train_test_df.columns)

    print("Number of null/inf in dataframe: %d" % train_test_df.isin([np.inf, -np.inf, np.nan]).sum().sum())

    

    new_feats = []

    for col in train_test_df.columns:

        # Square

        train_test_df[col+'^2'] = train_test_df[col].pow(2)

        

        # Cube

        train_test_df[col+'^3'] = train_test_df[col].pow(3)



        # 4th power

        train_test_df[col+'^4'] = train_test_df[col].pow(4)



        # Cumulative percentile (not normalized)

        train_test_df[col+'_cp'] = rankdata(train_test_df[col])



        # Cumulative normal percentile

        train_test_df[col+'_cnp'] = norm.cdf(train_test_df[col])

        

        new_feats.extend([col+'^2', col+'^3', col+'^4', col+'_cp', col+'_cnp'])

    

    for col in new_feats:

        train_test_df[col] = ((train_test_df[col] - train_test_df[col].mean()) / train_test_df[col].std())

    

    train_X_df = train_test_df.iloc[0:200000]

    test_X_df = train_test_df.iloc[200000:]

    del train_test_df; gc.collect()

    

    return train_X_df, test_X_df
%%time



# cat_params = {

#     'objective': 'Logloss',

#     'eval_metric': 'AUC',

#     'od_type': 'Iter',

#     'depth': 2,

#     'bootstrap_type': 'Bernoulli',

#     'random_seed': seed,

#     'allow_writing_files': False}



# model_name='cat'

# shuffle = True

# n_splits = 2

# num_round = 50000

# early_stop = 3000

# model_hpo = CVForBO(model=model_name, train_X=train_X, train_y=train_y, test_X=test_X, 

#                     base_params=cat_params, int_params=[], n_splits=n_splits, 

#                     num_round=num_round, early_stop=early_stop, shuffle=shuffle)



# cat_BO = BayesianOptimization(model_hpo.cv, {

#     'subsample': (0.2, 0.6), 

#     'l2_leaf_reg': (20, 100), 

#     'random_strength': (2, 20), 

#     'eta': (0.001, 0.02),

#     'colsample_bylevel': (0.03, 0.2),

#     }, random_state=seed)



# cat_BO = BayesianOptimization(model_hpo.cv, {

#     'l2_leaf_reg': (20, 100),

#     'random_strength': (2, 20),

#     'eta': (0.001, 0.02),

#     'subsample': (0.2, 0.6),

#     }, random_state=seed)



# cat_BO.maximize(init_points=8, n_iter=15)



# print(cat_BO.max)

# model_hpo.post_process()
%%time

# model_seeds_list = [1337, 99999, 2018, 516, 986, 6846, 654456, 357951, 17971, 55599]

model_seeds_list = [2018, 516, 986]



# ======================

# LightGBM

# ======================

# model_name = 'lgbm'

# scaler = False

# generate_features = False

# num_round = 1000000

# n_splits = 12

# shuffle = False

# early_stop = 3500

# lgbm_params = {

#     'bagging_freq': 5,

#     'bagging_fraction': 0.38,

#     'boost_from_average':'false',

#     'boost': 'gbdt',

#     'feature_fraction': 0.045,

#     'learning_rate': 0.0095,

#     'max_depth': -1,  

#     'metric':'auc',

#     'min_data_in_leaf': 80,

#     'min_sum_hessian_in_leaf': 10.0,

#     'num_leaves': 13,

#     'num_threads': 8,

#     'tree_learner': 'serial',

#     'objective': 'binary', 

#     'verbosity': 1,

#     'random_state': seed

# }  

    

# ======================

# CatBoost

# ======================

# model_name = 'cat'

# scaler = False

# generate_features = False

# num_round = 20000

# n_splits = 5

# early_stop = 500

# shuffle = True

# cat_params = {

#     'eval_metric': 'AUC',

#     'bootstrap_type': 'Bernoulli',

#     'objective': 'Logloss',

#     'od_type': 'Iter',

#     'depth': 2, 

#     'eta': 0.018, 

#     'l2_leaf_reg': 30, 

#     'random_strength': 5.41,

#     'random_seed': seed,

#     'allow_writing_files': False

# }



model_name = 'cat'

scaler = False

generate_features = False

num_round = 100000

n_splits = 12

early_stop = 2000

shuffle = True

        

cat_params = {

    'eval_metric': 'AUC',

    'bootstrap_type': 'Bernoulli',

    'objective': 'Logloss',

    'od_type': 'Iter',

    'depth': 2, 

    'eta': 0.0095, 

    'l2_leaf_reg': 30, 

    'random_strength': 5.41,

    'subsample': 0.4, 

    'random_seed': seed,

    'allow_writing_files': False,

}



# ======================

# GaussianNB

# ======================

# model_name = GaussianNB()

# scaler = 'qt'

# generate_features = False

# n_splits = 5

# shuffle = True



# ======================

# Logistic Regression

# ======================

# logreg_params = {

#     'tol': 0.0001, 

#     'C': 2.5, 

#     'random_state': seed, 

#     'max_iter': 5000,

#     'fit_intercept': True,

#     'solver': 'liblinear',

#     'multi_class': 'ovr',

#     'random_state': seed,

#     'verbose': 0

# }

# model_name = LogisticRegression(**logreg_params)

# scaler = False

# generate_features = True

# n_splits = 5

# early_stop = 500

# shuffle = True



# Generate features

if generate_features:

    train_X_df, test_X_df = features_generation(train_X, test_X)

else:

    train_X_df, test_X_df = train_X, test_X



# Transformer/Scaler

if scaler == 'ss':

    train_X_df, test_X_df = standard_scaler(train_X, test_X)

elif scaler == 'qt':

    train_X_df, test_X_df = quantile_transformer(train_X, test_X)

elif scaler == 'mm':

    train_X_df, test_X_df = min_max_scaler(train_X, test_X)

elif scaler == 'pt':

    train_X_df, test_X_df = power_transformer(train_X, test_X)

elif scaler == 'rs':

    train_X_df, test_X_df = robust_scaler(train_X, test_X)



# Classifier name (to be used when storing csv)

clf_name = model_name if isinstance(model_name, str) else model_name.__class__.__name__.lower()

clf_name = clf_name + '_' + scaler if scaler else clf_name

    

oofs = []

predictions = []

kfold_scores = {}

final_cv_score = {}

for i, seed_ in enumerate(model_seeds_list):

    i += 1

    print("\n#%d: processing seed %d" % (i, seed_))

    

    if model_name == 'cat':

        params = cat_params.copy()

        params['random_seed'] = seed_

    elif model_name == 'lgbm':

        params = lgbm_params.copy()   

        params['random_state'] = seed_

    elif model_name == GaussianNB():

        params = {}

    elif model_name == LogisticRegression():

        params = logreg_params.copy()   

        params['random_state'] = seed_

        model_name.set_params(**params)

    else: 

        params = {}

    

    model = CVClassifier(model_name, n_splits=n_splits, stratified=True, num_round=num_round, early_stop=early_stop, shuffle=shuffle, **params)

    model.cv(train_X_df, train_y)

    model.predict(test_X_df)



    kfold_scores[seed_] = model.scores_

    final_cv_score[seed_] = model.score_

    oofs.append(pd.DataFrame(np.array(model.oof_).T, columns=['seed_' + str(i)]))

    predictions.append(pd.DataFrame(np.array(model.predictions_).T, columns=['seed_' + str(i)]))

    

# Bagged OOF and test predictions

oof_df = pd.concat(oofs, axis=1)

oof_df = pd.concat([oof_df, train_y], axis=1)

oof_df.insert(loc=0, column=index_column, value=train_data[index_column].values)



predictions_df = pd.concat(predictions, axis=1)

predictions_df.insert(loc=0, column=index_column, value=test_data[index_column].values)



# CV scores

kfold_scores_df = pd.DataFrame([kfold_scores]).T.reset_index()

kfold_scores_df.rename(columns={'index': 'seed', 0: 'cv_score_per_each_fold'}, inplace=True)

kfold_scores_df.insert(loc=1, column='cv_std', value=kfold_scores_df['cv_score_per_each_fold'].map(

    lambda x: np.round(np.std(x), precision)))



final_cv_score_df = pd.DataFrame(index=list(final_cv_score.keys()), data=list(final_cv_score.values()), 

                                 columns=['cv_mean_score']).reset_index()

final_cv_score_df.rename(columns={'index': 'seed'}, inplace=True)

final_cv_score_df['cv_mean_score'] = final_cv_score_df['cv_mean_score'].map(lambda x: np.round(x, precision))

final_cv_score_df = final_cv_score_df.merge(kfold_scores_df, how='left', on='seed')



# Saving data

oof_df.to_csv(clf_name + '_train_OOF_bagged.csv', index=False)

predictions_df.to_csv(clf_name + '_test_bagged.csv', index=False)

final_cv_score_df.to_csv(clf_name + '_cv_results.csv', index=False)



print('\nBagged CV scores: %s' % list(map(lambda x: np.round(x, precision), list(final_cv_score.values()))))

print('Bagged mean CV: {0} +/- {1}\n'.format(np.round(np.mean(list(final_cv_score.values())), precision), 

                                             np.round(np.std(list(final_cv_score.values())), precision)))
# ================================

# LightGBM

# ================================

# 12 Folds

# KFold CV: [0.90147, 0.89969, 0.89186, 0.90704, 0.89463, 0.90097, 0.90226, 0.90352, 0.8987, 0.90397, 0.90671, 0.8996]

# Mean Kfold CV: 0.90087 +/- 0.00427

# Estimated CV: 0.90077

# Wall time: 95 min



# # 12 Folds 3 seeds

# Bagged CV scores: [0.90077, 0.90029, 0.90017]

# Bagged mean CV: 0.90041 +/- 0.00026

# Wall time: 6h 23min 10s
# ================================

# Linear Regression

# ================================

# 2 folds, 'tol': 0.0001, 'C': 1.0

# KFold CV: [0.89664, 0.89267]

# Mean Kfold CV: 0.89466 +/- 0.00198

# Estimated CV: 0.89463

# Wall time: 10min 10s



# 2 folds, 'tol': 0.0001, 'C': 2.0

# KFold CV: [0.89667, 0.89272]

# Mean Kfold CV: 0.8947 +/- 0.00198

# Estimated CV: 0.89467

# Wall time: 10min 8s



# 2 folds, 'tol': 0.0001, 'C': 3.0

# KFold CV: [0.89665, 0.89273]

# Mean Kfold CV: 0.89469 +/- 0.00196

# Estimated CV: 0.89466

# Wall time: 12min 18s
# ================================

# Gaussian Naive Bayes

# ================================

# standard_scaler

# KFold CV: [0.89241, 0.88971, 0.88199, 0.89609, 0.89236, 0.8892, 0.88226, 0.88616, 0.88875, 0.88533]

# Mean Kfold CV: 0.88843 +/- 0.00432

# Estimated CV: 0.88844



# quantile_transformer

# KFold CV: [0.8933, 0.89025, 0.88326, 0.89701, 0.89259, 0.8903, 0.88379, 0.88746, 0.88942, 0.88624]

# Mean Kfold CV: 0.88936 +/- 0.00409

# Estimated CV: 0.88937



# min_max_scaler

# KFold CV: [0.89241, 0.88971, 0.88199, 0.89609, 0.89236, 0.8892, 0.88226, 0.88616, 0.88875, 0.88533]

# Mean Kfold CV: 0.88843 +/- 0.00432

# Estimated CV: 0.88844

    

# power_transformer

# KFold CV: [0.89225, 0.88938, 0.88188, 0.89513, 0.89155, 0.88827, 0.88183, 0.88573, 0.88782, 0.88494]

# Mean Kfold CV: 0.88788 +/- 0.00417

# Estimated CV: 0.88789

    

# robust_scaler

# KFold CV: [0.89241, 0.88971, 0.88199, 0.89609, 0.89236, 0.8892, 0.88226, 0.88616, 0.88875, 0.88533]

# Mean Kfold CV: 0.88843 +/- 0.00432

# Estimated CV: 0.88844
predictions_df_mean = predictions_df.copy()

pred_cols = predictions_df_mean.columns

predictions_df_mean[target_column] = predictions_df_mean.loc[:, ~predictions_df_mean.columns.isin([index_column, target_column])].apply(np.mean, axis=1)  # 0 column is index

predictions_df_mean = predictions_df_mean[[index_column, target_column]]

predictions_df_mean.to_csv(clf_name + '_test.csv', index=False)