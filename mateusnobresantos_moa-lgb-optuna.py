import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')



train.shape
train_target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



train_target.shape
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



test.shape
antagonists = []



for column in train_target.columns:

    if column.endswith('_agonist'):

        antagonist = column.replace('_agonist', '_antagonist')

        if antagonist in train_target.columns:

            antagonists.append((column, antagonist))



antagonists
for pair in antagonists:

    n = train_target[(train_target[pair[0]] == 1) & (train_target[pair[1]] == 1)].shape[0]

    if n > 0:

        print(pair[0], '-', pair[1])

        print('Number of cases:', n)
# From https://www.kaggle.com/carlmcbrideellis/moa-setting-ctl-vehicle-0-improves-score



train.at[train['cp_type'].str.contains('ctl_vehicle'),train.filter(regex='-.*').columns] = 0.0



test.at[test['cp_type'].str.contains('ctl_vehicle'),test.filter(regex='-.*').columns] = 0.0
train_size = train.shape[0]



traintest = pd.concat([train, test])



traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_type'], prefix='cp_type')], axis=1)

traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_time'], prefix='cp_time')], axis=1)

traintest = pd.concat([traintest, pd.get_dummies(traintest['cp_dose'], prefix='cp_dose')], axis=1)



traintest = traintest.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)



train = traintest[:train_size]

test  = traintest[train_size:]



del traintest
train.shape
x_train = train.drop('sig_id', axis=1)



y_train = train_target.drop('sig_id', axis=1)



x_test = test.drop('sig_id', axis=1)
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import log_loss



import lightgbm as lgb
def fit_predict(n_splits, params, x_train, y_train, x_test):

    

    oof = np.zeros(x_train.shape[0])

    

    y_preds = []

    

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        

        x_train_train = x_train.iloc[train_idx]

        y_train_train = y_train.iloc[train_idx]

        x_train_valid = x_train.iloc[valid_idx]

        y_train_valid = y_train.iloc[valid_idx]



        lgb_train = lgb.Dataset(data=x_train_train.astype('float32'), label=y_train_train.astype('float32'))

        lgb_valid = lgb.Dataset(data=x_train_valid.astype('float32'), label=y_train_valid.astype('float32'))



        estimator = lgb.train(params, lgb_train, 10000, valid_sets=lgb_valid,

                              early_stopping_rounds=25, verbose_eval=0)



        oof_part = estimator.predict(x_train_valid, num_iteration=estimator.best_iteration)

        oof[valid_idx] = oof_part

        

        if x_test is not None:

            y_part = estimator.predict(x_test, num_iteration=estimator.best_iteration)

            y_preds.append(y_part)

        

    score = log_loss(y_train, oof)

    print('LogLoss Score:', score)

    

    y_pred = np.mean(y_preds, axis=0)

    

    return y_pred, oof, score
import optuna





columns_to_try = [

    'glutamate_receptor_antagonist',

    'dna_inhibitor',

    'serotonin_receptor_antagonist',

    'dopamine_receptor_antagonist',

    'cyclooxygenase_inhibitor'

]



def objective(trial):

    

    params = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'boosting_type': 'gbdt',

        'boost_from_average': True,

        'num_threads': 4,

        'random_state': 42,

        

        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),

        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),

        'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.001, 0.1),

        'max_depth': trial.suggest_int('max_depth', 1, 100),

        'bagging_fraction': trial.suggest_loguniform('bagging_fraction', .5, .99),

        'feature_fraction': trial.suggest_loguniform('feature_fraction', .5, .99),

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 0.1, 2),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 0.1, 2)

    }

    

    scores = []

    for column in columns_to_try:

        _, _, score = fit_predict(3, params, x_train, y_train[column], None)

        scores.append(score)

    

    return np.mean(scores)





# study = optuna.create_study(direction='minimize')

# study.optimize(objective, n_trials=100)
# study.best_trial
columns_1 = [

    '5-alpha_reductase_inhibitor',

    '11-beta-hsd1_inhibitor',

    'adenylyl_cyclase_activator',

    'aldehyde_dehydrogenase_inhibitor',

    'ampk_activator',

    'analgesic',

    'antiarrhythmic',

    'anticonvulsant',

    'antifungal',

    'antihistamine',

    'antimalarial',

    'antiviral',

    'atm_kinase_inhibitor',

    'atp-sensitive_potassium_channel_antagonist',

    'atp_synthase_inhibitor',

    'atr_kinase_inhibitor',

    'autotaxin_inhibitor',

    'bacterial_membrane_integrity_inhibitor',

    'calcineurin_inhibitor',

    'caspase_activator',

    'catechol_o_methyltransferase_inhibitor',

    'cck_receptor_antagonist',

    'chk_inhibitor',

    'coagulation_factor_inhibitor',

    'diuretic',

    'elastase_inhibitor',

    'erbb2_inhibitor',

    'farnesyltransferase_inhibitor',

    'focal_adhesion_kinase_inhibitor',

    'free_radical_scavenger',

    'fungal_squalene_epoxidase_inhibitor',

    'glutamate_inhibitor',

    'gonadotropin_receptor_agonist',

    'histone_lysine_demethylase_inhibitor',

    'hsp_inhibitor',

    'ikk_inhibitor',

    'laxative',

    'leukotriene_inhibitor',

    'lipase_inhibitor',

    'lxr_agonist',

    'mdm_inhibitor',

    'monoacylglycerol_lipase_inhibitor',

    'monopolar_spindle_1_kinase_inhibitor',

    'nicotinic_receptor_agonist',

    'nitric_oxide_production_inhibitor',

    'norepinephrine_reuptake_inhibitor',

    'nrf2_activator',

    'pdk_inhibitor',

    'progesterone_receptor_antagonist',

    'proteasome_inhibitor',

    'protein_phosphatase_inhibitor',

    'protein_tyrosine_kinase_inhibitor',

    'ras_gtpase_inhibitor',

    'retinoid_receptor_antagonist',

    'steroid',

    'syk_inhibitor',

    'tgf-beta_receptor_inhibitor',

    'thrombin_inhibitor',

    'tlr_antagonist',

    'transient_receptor_potential_channel_antagonist',

    'tropomyosin_receptor_kinase_inhibitor',

    'trpv_agonist',

    'ubiquitin_specific_protease_inhibitor',

    'vitamin_d_receptor_agonist'

]
params_1 = {

    'objective': 'binary',

    'metric': 'binary_logloss',

    'boosting_type': 'gbdt',

    'boost_from_average': True,

    'num_threads': 4,

    'random_state': 42,

    

    'learning_rate': 0.01,

    

    # from Optuna result in Version 7

    'num_leaves': 212,

    'min_data_in_leaf': 92,

    'min_child_weight': 0.0010123391323415569,

    'max_depth': 35,

    'bagging_fraction': 0.7968351296815959,

    'feature_fraction': 0.7556374471450119,

    'lambda_l1': 0.23497601594060086,

    'lambda_l2': 0.15889208239516134

}



params_1
params_2 = {

    'objective': 'binary',

    'metric': 'binary_logloss',

    'boosting_type': 'gbdt',

    'boost_from_average': True,

    'num_threads': 4,

    'random_state': 42,

    

    'learning_rate': 0.01,

    

    # from Optuna result in Version 15

    'num_leaves': 106,

    'min_data_in_leaf': 176,

    'min_child_weight': 0.08961015929882983,

    'max_depth': 3,

    'bagging_fraction': 0.5672004837454858,

    'feature_fraction': 0.611628226420641,

    'lambda_l1': 1.293005852529098,

    'lambda_l2': 1.6012450757049599

}



params_2
n_splits = 3



y_pred = pd.DataFrame()



oof = pd.DataFrame()



scores = []



for column in y_train.columns:

    print('Column:', column)

    

    if column in columns_1:

        print('Params 1')

        params = params_1

    else:

        print('Params 2')

        params = params_2

    

    y_pred[column], oof[column], score = fit_predict(n_splits, params, x_train, y_train[column], x_test)

    

    scores.append(score)
np.mean(scores)
x_train_2 = pd.concat([x_train, oof], axis=1)



x_train_2.shape
x_test_2 = pd.concat([x_test, y_pred], axis=1)



x_test_2.shape
import optuna





columns_to_try = [

    'antiviral',

    'dna_inhibitor'

]



def objective(trial):

    

    params = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'boosting_type': 'gbdt',

        'boost_from_average': True,

        'num_threads': 4,

        'random_state': 42,

        

        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),

        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),

        'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.001, 0.1),

        'max_depth': trial.suggest_int('max_depth', 1, 100),

        'bagging_fraction': trial.suggest_loguniform('bagging_fraction', .5, .99),

        'feature_fraction': trial.suggest_loguniform('feature_fraction', .5, .99),

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 0.1, 2),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 0.1, 2)

    }

    

    scores = []

    for column in columns_to_try:

        _, _, score = fit_predict(3, params, x_train_2.drop(column, axis=1), y_train[column], None)

        scores.append(score)

    

    return np.mean(scores)





# study = optuna.create_study(direction='minimize')

# study.optimize(objective, n_trials=100)
params = {

    'objective': 'binary',

    'metric': 'binary_logloss',

    'boosting_type': 'gbdt',

    'boost_from_average': True,

    'num_threads': 4,

    'random_state': 42,

    

    'learning_rate': 0.01,



    # from Optuna result in Version 21

    'num_leaves': 204,

    'min_data_in_leaf': 109,

    'min_child_weight': 0.0022436199989295377,

    'max_depth': 3,

    'bagging_fraction': 0.8715885202491637,

    'feature_fraction': 0.6187304356714799,

    'lambda_l1': 1.4290015875633637,

    'lambda_l2': 0.7289605844855718

}



params
y_pred_2 = pd.DataFrame()



oof_2 = pd.DataFrame()



scores_2 = []



for column in y_train.columns:

    print('Column:', column)

    

    y_pred_2[column], oof_2[column], score = fit_predict(n_splits, params, x_train_2.drop(column, axis=1),

                                                         y_train[column], x_test_2.drop(column, axis=1))

    

    scores_2.append(score)
y_pred = y_pred_2



oof = oof_2



scores = scores_2
np.mean(scores)
score = pd.DataFrame()

score['feature'] = y_train.columns

score['score'] = scores + [0] * (len(y_train.columns) - len(scores))
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10,40))



sns.barplot(x="score", y="feature", data=score)



plt.show()
for pair in antagonists:

    n = y_pred[(y_pred[pair[0]] > 0.5) & (y_pred[pair[1]] > 0.5)].shape[0]

    if n > 0:

        print(pair[0], '-', pair[1])

        print('Number of cases:', n)
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



columns = list(set(submission.columns) & set(y_pred.columns))

submission[columns] = y_pred[columns]



submission.to_csv('submission.csv', index=False)
oof.to_csv('oof.csv', index=False)



score.to_csv('score.csv', index=False)