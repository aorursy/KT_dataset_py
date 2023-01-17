# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
x_train = train.drop('sig_id', axis=1)



y_train = train_target.drop('sig_id', axis=1)



x_test = test.drop('sig_id', axis=1)
oof = pd.read_csv('/kaggle/input/moa-lgb-optuna/oof.csv')

preds = pd.read_csv('/kaggle/input/moa-lgb-optuna/preds.csv')
from sklearn.metrics import roc_auc_score, log_loss

scores = []

for col in y_train.columns:

    auc = roc_auc_score(y_train[col], oof[col])

    loss = log_loss(y_train[col], oof[col])

    loss_clip = log_loss(y_train[col], np.clip(oof[col], 0.025, 0.975))

    scores.append([col, auc, loss, loss_clip])

score_df = pd.DataFrame(scores, columns=['col', 'auc', 'loss', 'loss_clip'])

score_df
score_df['loss'].mean()
def mean_log_loss(c, y_true, y_pred):

    scores = []

    for col in y_true.columns:

#         auc = roc_auc_score(y_true[col], y_pred[col])

#         loss = log_loss(y_train[col], y_pred[col])

        loss_clip = log_loss(y_true[col], np.clip(y_pred[col], c, 1-c))

        scores.append([col, loss_clip])

    score_df = pd.DataFrame(scores, columns=['col','loss_clip'])

    return score_df['loss_clip'].mean()
## Find optimal clipping threshold

clip_scores = []

for c in np.arange(0.001, 0.02, 0.001):

    clip_scores.append([c, mean_log_loss(c, y_train, oof)])

clip_score_df = pd.DataFrame(clip_scores,columns=['threshold', 'loss'])

clip_score_df
import seaborn as sns

sns.distplot(score_df['auc'])
sns.distplot(score_df['loss'])
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
train_x
## Append oof to features

x_train = pd.concat((x_train, oof), axis=1)

x_test = pd.concat((x_test, preds), axis=1)



print(x_train.shape, x_test.shape)
pd.concat((x_train, oof), axis=1)
n_splits = 3



y_pred = pd.DataFrame()



oof = pd.DataFrame()



scores = []



for i, column in enumerate(y_train.columns):

    print('Column:', column, '#', i)

    

    if column in columns_1:

        print('Params 1')

        params = params_1

    else:

        print('Params 2')

        params = params_2

    

    y_pred[column], oof[column], score = fit_predict(n_splits, params, x_train, y_train[column], x_test)

    

    scores.append(score)
oof.to_csv('oof_l2.csv', index=False)

y_pred.to_csv('preds_l2.csv', index=False)
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



columns = list(set(submission.columns) & set(y_pred.columns))

submission[columns] = y_pred[columns]



submission.to_csv('submission.csv', index=False)
from sklearn.metrics import roc_auc_score, log_loss

scores = []

for col in y_train.columns:

    auc = roc_auc_score(y_train[col], oof[col])

    loss = log_loss(y_train[col], oof[col])

    loss_clip = log_loss(y_train[col], np.clip(oof[col], 0.025, 0.975))

    scores.append([col, auc, loss, loss_clip])

score_df = pd.DataFrame(scores, columns=['col', 'auc', 'loss', 'loss_clip'])

score_df
# Check performance

print(score_df['auc'].mean(), score_df['loss'].mean(), score_df['loss_clip'].mean())
## Find optimal clipping threshold

clip_scores = []

for c in np.arange(0.001, 0.02, 0.001):

    clip_scores.append([c, mean_log_loss(c, y_train, oof)])

clip_score_df = pd.DataFrame(clip_scores,columns=['threshold', 'loss'])

clip_score_df
print(clip_score_df.sort_values('loss')['loss'].values[0])

threshold = clip_score_df.sort_values('loss')['threshold'].values[0]

print(threshold)
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



columns = list(set(submission.columns) & set(y_pred.columns))

submission[columns] = np.clip(y_pred[columns], threshold, 1-threshold)



submission.to_csv('submission_clipped.csv', index=False)