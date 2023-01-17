import os

import gc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

train.head(10)
print(test.shape)

test.head(10)
target_fe = np.log1p(train.formation_energy_ev_natom)

target_be = np.log1p(train.bandgap_energy_ev)

del train['formation_energy_ev_natom'], train['bandgap_energy_ev'], train['id'], test['id']
sorted(train['spacegroup'].unique())
sorted(test['spacegroup'].unique())
train = pd.concat([train.drop(['spacegroup'], axis=1), 

                   pd.get_dummies(train['spacegroup'], prefix='SG')], axis=1)

test = pd.concat([test.drop(['spacegroup'], axis=1), 

                   pd.get_dummies(test['spacegroup'], prefix='SG')], axis=1)
import lightgbm as lgb

import multiprocessing



def cv_train_model(X, y, 

                   verbose_eval=None, 

                   early_stopping_rounds=None,

                   params=None):

    if type(y) is pd.core.frame.DataFrame:

        y = y.values.ravel()

    dstrain = lgb.Dataset(X, label=y)

    max_boost_round = 4000

    if params is None:

        lgb_params = {

            'objective': 'regression_l2',

            'learning_rate': 0.008,

            'num_threads': 4,#multiprocessing.cpu_count(),

            'max_depth': 4,

            'min_data_in_leaf': 23,

            'feature_fraction': 0.93,

            'bagging_fraction': 0.93,

            'bagging_freq': 1,

            'lambda_l2': 1e2,

            'metric': ['mse']

        }

    print('lgb cv and training...')

    if verbose_eval is None:

        verbose_eval = int(max_boost_round/30)

    if early_stopping_rounds is None:

        early_stopping_rounds = int(max_boost_round/10)

    cv_lgb = lgb.cv(lgb_params, dstrain,

                    num_boost_round=max_boost_round,

                    nfold=10,

                    stratified=False,

                    verbose_eval=verbose_eval,

                    early_stopping_rounds=early_stopping_rounds,

                    show_stdv=False)

    best_round = np.argmin(cv_lgb['l2-mean'])

    best_cv_mean = np.min(cv_lgb['l2-mean'])

    print('best round', best_round)

    print('best mse-mean', best_cv_mean)

    model_lgb = lgb.train(lgb_params, dstrain, 

                          num_boost_round=best_round,

                          valid_sets=dstrain,

                          verbose_eval=verbose_eval)

    print('lgb cv and training finished...')

    return model_lgb, best_cv_mean

def get_feat_weight(model_lgb, feat_names, plot=True):

    feat_weight = pd.DataFrame(model_lgb.feature_importance(),

                               columns=['feature_importance'],

                               index=feat_names)

    if plot:

        indices = np.argsort(feat_weight['feature_importance'])[::-1]

        plt.figure(figsize=(12, 6))

        plt.title('feature importance (lightgbm)')

        plt.bar(range(len(feat_weight)), list(feat_weight.iloc[indices, 0]))

        plt.xticks(range(len(feat_weight)), feat_weight.iloc[indices].index, 

                   rotation='vertical')

        plt.xlim([-1, len(feat_weight)])

        plt.show()

    return feat_weight

def get_model_cv(df, y, plot=True, verbose_eval=False):

    model_lgb, best_cv_mean = cv_train_model(df, y, verbose_eval=verbose_eval)

    feat_weight = get_feat_weight(model_lgb, feat_names=df.columns, plot=plot)

    print('best cv mean', best_cv_mean)

    return best_cv_mean, feat_weight, model_lgb
best_cv_mean_fe, feat_weight_fe, model_lgb_fe = get_model_cv(train, target_fe)

pred_fe = np.expm1(model_lgb_fe.predict(test))

best_cv_mean_be, feat_weight_be, model_lgb_be = get_model_cv(train, target_be)

pred_be = np.expm1(model_lgb_be.predict(test))

scr_total = np.mean([np.sqrt(best_cv_mean_fe), np.sqrt(best_cv_mean_be)])

print(f'total cv score: {scr_total}')

sub = pd.read_csv('../input/sample_submission.csv')

sub['formation_energy_ev_natom'] = pred_fe

sub['bandgap_energy_ev'] = pred_be

sub.to_csv(f'sb_{scr_total}.csv', index=False) ### LB ~0.0570
### uncomment to get cv pred

from sklearn.linear_model import RidgeCV

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

rg = RidgeCV(alphas=[0.003, 0.01, 0.3, 3, 10], cv=5)



def kfold_cv(X, y, test, n_splits=10, 

             train_lgb=True, 

             lgb_ratio=0.8,

             cv_pred_test=False):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=233)

    cv_pred = np.zeros((len(test)))

    scr_total = 0

    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(y)):

        print(f'Starting No.{fold_id} fold CV out of {n_splits} ...')

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]

        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        rg.fit(StandardScaler().fit_transform(X_tr), y_tr)

        pred_rg = rg.predict(StandardScaler().fit_transform(X_te))

        mse = mean_squared_error(y_te, pred_rg)

        print('=======rg mse:', mse)

        if train_lgb==True:

            _, _, model_lgb = get_model_cv(X_tr, y_tr, False, 0)

            print('=======lgb mse:',mean_squared_error(y_te, model_lgb.predict(X_te)))

            avg_mse = mean_squared_error(y_te, 

                                         pred_rg*(1-lgb_ratio)+\

                                         model_lgb.predict(X_te)*lgb_ratio)

            print(f'=======avg mse: {avg_mse}')

            scr_total += avg_mse / n_splits

        else:

            scr_total += mse / n_splits

        if cv_pred_test == True:

            if train_lgb == True: 

                cv_pred += (rg.predict(StandardScaler().fit_transform(

                        test))*(1-lgb_ratio) + \

                            model_lgb.predict(test)*lgb_ratio)/n_splits

            else:

                cv_pred += rg.predict(StandardScaler().fit_transform(

                        test))/n_splits

    print(f'score total: {scr_total}')

    if not cv_pred_test:

        return scr_total

    else:

        return scr_total, np.expm1(cv_pred)

#lgb_ratio = 0.95

#cv_fe, cv_pred_fe = kfold_cv(train, target_fe, test, n_splits=10, 

#                             train_lgb=True, lgb_ratio=lgb_ratio, cv_pred_test=True)

#cv_be, cv_pred_be = kfold_cv(train, target_be, test, n_splits=10, 

#                             train_lgb=True, lgb_ratio=lgb_ratio, cv_pred_test=True)

#cv_total = np.mean([np.sqrt(cv_fe), np.sqrt(cv_be)])

#print(f'fe: {cv_fe}; be: {cv_be}')

#print(f'cv total rmsle:{cv_total}')

#sub['formation_energy_ev_natom'] = cv_pred_fe

#sub['bandgap_energy_ev'] = cv_pred_be

#sub.to_csv(f'sb_lgb{lgb_ratio}_rg_{cv_total}.csv', index=False) ### LB ~0.0573