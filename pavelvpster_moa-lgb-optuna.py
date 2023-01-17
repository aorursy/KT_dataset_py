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
# from https://www.kaggle.com/pavelvpster/moa-keras-optuna-rfecv



# 500

selected_features = ['g-0', 'g-1', 'g-2', 'g-3', 'g-4', 'g-5', 'g-6', 'g-7', 'g-8', 'g-9', 'g-10', 'g-11', 'g-12', 'g-13', 'g-14', 'g-15', 'g-16', 'g-17', 'g-18', 'g-19', 'g-20', 'g-21', 'g-22', 'g-23', 'g-24', 'g-25', 'g-26', 'g-27', 'g-28', 'g-29', 'g-30', 'g-31', 'g-32', 'g-33', 'g-34', 'g-35', 'g-36', 'g-37', 'g-38', 'g-39', 'g-40', 'g-41', 'g-42', 'g-43', 'g-44', 'g-45', 'g-46', 'g-47', 'g-48', 'g-49', 'g-50', 'g-51', 'g-52', 'g-53', 'g-54', 'g-55', 'g-56', 'g-57', 'g-58', 'g-59', 'g-60', 'g-61', 'g-62', 'g-63', 'g-64', 'g-65', 'g-66', 'g-67', 'g-68', 'g-69', 'g-70', 'g-71', 'g-72', 'g-73', 'g-74', 'g-75', 'g-76', 'g-77', 'g-78', 'g-79', 'g-80', 'g-81', 'g-82', 'g-83', 'g-84', 'g-85', 'g-86', 'g-87', 'g-88', 'g-89', 'g-90', 'g-91', 'g-92', 'g-93', 'g-94', 'g-95', 'g-96', 'g-97', 'g-98', 'g-99', 'g-100', 'g-101', 'g-102', 'g-103', 'g-104', 'g-105', 'g-106', 'g-107', 'g-108', 'g-109', 'g-110', 'g-111', 'g-112', 'g-113', 'g-114', 'g-115', 'g-116', 'g-117', 'g-118', 'g-121', 'g-122', 'g-123', 'g-125', 'g-126', 'g-127', 'g-128', 'g-129', 'g-130', 'g-131', 'g-132', 'g-133', 'g-139', 'g-144', 'g-149', 'g-152', 'g-153', 'g-154', 'g-155', 'g-158', 'g-161', 'g-162', 'g-165', 'g-166', 'g-167', 'g-168', 'g-169', 'g-170', 'g-171', 'g-172', 'g-173', 'g-174', 'g-175', 'g-176', 'g-177', 'g-178', 'g-179', 'g-180', 'g-182', 'g-185', 'g-188', 'g-189', 'g-191', 'g-192', 'g-193', 'g-194', 'g-195', 'g-196', 'g-197', 'g-198', 'g-199', 'g-200', 'g-201', 'g-202', 'g-203', 'g-204', 'g-209', 'g-211', 'g-213', 'g-214', 'g-217', 'g-220', 'g-221', 'g-222', 'g-223', 'g-225', 'g-226', 'g-229', 'g-233', 'g-239', 'g-240', 'g-243', 'g-244', 'g-246', 'g-247', 'g-250', 'g-254', 'g-255', 'g-256', 'g-257', 'g-259', 'g-261', 'g-262', 'g-265', 'g-269', 'g-270', 'g-271', 'g-273', 'g-279', 'g-282', 'g-284', 'g-286', 'g-289', 'g-292', 'g-294', 'g-295', 'g-298', 'g-299', 'g-300', 'g-303', 'g-306', 'g-310', 'g-311', 'g-312', 'g-314', 'g-316', 'g-317', 'g-318', 'g-320', 'g-321', 'g-323', 'g-326', 'g-327', 'g-328', 'g-332', 'g-333', 'g-335', 'g-339', 'g-340', 'g-341', 'g-346', 'g-347', 'g-351', 'g-353', 'g-355', 'g-357', 'g-360', 'g-364', 'g-365', 'g-370', 'g-375', 'g-383', 'g-385', 'g-388', 'g-389', 'g-390', 'g-391', 'g-392', 'g-399', 'g-401', 'g-402', 'g-403', 'g-404', 'g-409', 'g-417', 'g-423', 'g-424', 'g-425', 'g-426', 'g-428', 'g-429', 'g-430', 'g-431', 'g-433', 'g-434', 'g-438', 'g-439', 'g-440', 'g-441', 'g-443', 'g-444', 'g-445', 'g-446', 'g-447', 'g-448', 'g-449', 'g-450', 'g-451', 'g-452', 'g-453', 'g-454', 'g-455', 'g-456', 'g-460', 'g-461', 'g-463', 'g-465', 'g-467', 'g-470', 'g-473', 'g-474', 'g-475', 'g-476', 'g-481', 'g-483', 'g-485', 'g-486', 'g-489', 'g-490', 'g-495', 'g-500', 'g-501', 'g-502', 'g-504', 'g-509', 'g-513', 'g-516', 'g-517', 'g-518', 'g-527', 'g-531', 'g-535', 'g-540', 'g-542', 'g-543', 'g-544', 'g-547', 'g-548', 'g-549', 'g-552', 'g-556', 'g-558', 'g-562', 'g-566', 'g-571', 'g-572', 'g-573', 'g-574', 'g-578', 'g-584', 'g-585', 'g-590', 'g-592', 'g-595', 'g-596', 'g-600', 'g-602', 'g-603', 'g-605', 'g-606', 'g-611', 'g-612', 'g-615', 'g-617', 'g-618', 'g-619', 'g-620', 'g-623', 'g-624', 'g-625', 'g-626', 'g-627', 'g-628', 'g-629', 'g-630', 'g-631', 'g-632', 'g-633', 'g-634', 'g-635', 'g-636', 'g-637', 'g-638', 'g-639', 'g-640', 'g-643', 'g-645', 'g-646', 'g-648', 'g-651', 'g-654', 'g-657', 'g-658', 'g-659', 'g-660', 'g-661', 'g-662', 'g-663', 'g-664', 'g-665', 'g-666', 'g-668', 'g-670', 'g-672', 'g-673', 'g-674', 'g-675', 'g-677', 'g-678', 'g-679', 'g-685', 'g-689', 'g-695', 'g-697', 'g-698', 'g-699', 'g-701', 'g-702', 'g-703', 'g-704', 'g-707', 'g-709', 'g-710', 'g-712', 'g-713', 'g-714', 'g-715', 'g-721', 'g-722', 'g-724', 'g-725', 'g-726', 'g-734', 'g-736', 'g-737', 'g-740', 'g-742', 'g-743', 'g-745', 'g-747', 'g-750', 'g-751', 'g-752', 'g-753', 'g-756', 'g-757', 'g-758', 'g-759', 'g-760', 'g-761', 'g-762', 'g-763', 'g-764', 'g-766', 'g-767', 'g-768', 'g-769', 'g-770', 'g-771', 'c-0', 'c-1', 'c-2', 'c-3', 'c-4', 'c-5', 'c-11', 'c-12', 'c-14', 'c-19', 'c-20', 'c-21', 'c-22', 'c-23', 'c-25', 'c-27', 'c-32', 'c-35', 'c-36', 'c-37', 'c-40', 'c-47', 'c-54', 'c-55', 'c-57', 'c-58', 'c-61', 'c-62', 'c-63', 'c-65', 'c-66', 'c-71', 'c-74', 'c-75', 'c-77', 'c-78', 'c-79', 'c-80', 'c-81', 'c-82', 'c-83', 'c-84', 'c-85', 'c-86', 'c-87', 'c-88', 'c-89', 'c-90', 'c-91', 'c-92', 'c-93', 'c-97']
x_train = x_train[selected_features]



x_test = x_test[selected_features]
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