import os
import random
import platform
import itertools
import gc

import sklearn
import pandas as pd
import numpy as np
import lightgbm as lgb
print('Python version:', platform.python_version())
print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)
print('Scikit-Learn version:', sklearn.__version__)
print('LightGBM version:', lgb.__version__)
SEED = 42

os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
df_train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
df_train
df_train2 = pd.read_csv('/kaggle/input/melanomaextendedtabular/external_upsampled_tabular.csv')
df_train2
df_test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
df_test
df_train.isnull().sum()
df_train2.isnull().sum()
df_test.isnull().sum()
df_train['sex'].unique()
df_train2['sex'].unique()
df_test['sex'].unique()
df_train['anatom_site_general_challenge'].unique()
df_train2['anatom_site_general_challenge'].unique()
df_test['anatom_site_general_challenge'].unique()
# get image name to create submission file
image_name_test = df_test['image_name']
def process_sex(sex):
    if sex == 'male':
        return 1
    elif sex == 'female':
        return 0
    else:
        return np.nan
    
df_train['sex'] = df_train['sex'].apply(process_sex)
df_train2['sex'] = df_train2['sex'].apply(process_sex)
df_test['sex'] = df_test['sex'].apply(process_sex)
def process_site(site):
    if site == 'head/neck':
        return 0
    elif site == 'upper extremity':
        return 1
    elif site == 'lower extremity':
        return 2
    elif site == 'torso':
        return 3
    elif site == 'palms/soles':
        return 4
    elif site == 'oral/genital':
        return 5
    else:
        return np.nan
    
df_train['anatom_site_general_challenge'] = df_train['anatom_site_general_challenge'].apply(process_site)
df_train2['anatom_site_general_challenge'] = df_train2['anatom_site_general_challenge'].apply(process_site)
df_test['anatom_site_general_challenge'] = df_test['anatom_site_general_challenge'].apply(process_site)
image_count_train = df_train['patient_id'].value_counts()
image_count_test = df_test['patient_id'].value_counts()

df_train['image_count'] = df_train['patient_id'].apply(lambda pid: image_count_train[pid])
df_train2['image_count'] = pd.Series([np.nan for _ in range(df_train2.shape[0])])
df_test['image_count'] = df_test['patient_id'].apply(lambda pid: image_count_test[pid])
age_min_train = df_train.groupby('patient_id').age_approx.min()
age_max_train = df_train.groupby('patient_id').age_approx.max()
age_span_train = age_max_train - age_min_train
df_train['age_min'] = df_train['patient_id'].apply(lambda pid: age_min_train[pid])
df_train['age_max'] = df_train['patient_id'].apply(lambda pid: age_max_train[pid])
df_train['age_span'] = df_train['patient_id'].apply(lambda pid: age_span_train[pid])

df_train2['age_min'] = df_train['age_approx']
df_train2['age_max'] = df_train['age_approx']
df_train2['age_span'] = pd.Series([0 for _ in range(df_train2.shape[0])])

age_min_test = df_test.groupby('patient_id').age_approx.min()
age_max_test = df_test.groupby('patient_id').age_approx.max()
age_span_test = age_max_test - age_min_test
df_test['age_min'] = df_test['patient_id'].apply(lambda pid: age_min_test[pid])
df_test['age_max'] = df_test['patient_id'].apply(lambda pid: age_max_test[pid])
df_test['age_span'] = df_test['patient_id'].apply(lambda pid: age_span_test[pid])
# delete unused column
del df_train['image_name']
del df_train['patient_id']
del df_train['diagnosis']
del df_train['benign_malignant']

del df_train2['image_name']
del df_train2['width']
del df_train2['height']

del df_test['image_name']
del df_test['patient_id']
# get index of categorical feature
cat_feature = ['sex', 'anatom_site_general_challenge']
cat_feature_idx = [df_train.columns.get_loc(ct) for ct in cat_feature]
cat_feature_idx
# split to X and y
X = pd.concat([df_train, df_train2], axis=0).reset_index(drop=True)
y = X['target']
del X['target']

X_test = df_test.copy()
param_dict = {
    'boosting_type': ['gbdt', 'dart'],
    'learning_rate': [0.1, 0.03, 0.01],
    'n_estimators': [100, 300],
    'feature_fraction': [5/7 + 0.01, 1.0],
    'lambda': [
        # l1, l2
        [0.0, 0.0],
        [0.001, 0.01],
        [0.01, 0.1],
        [1.0, 0.01],
    ],
}
param_key = list(param_dict.keys())
param_item = list(param_dict.values())
param_item
param_list = list(itertools.product(*param_item))
param_list[:10]
len(param_list)
df_model = pd.DataFrame(columns=[*param_key, *[f'model_{i}' for i in range(5)], *[f'model_{i}_auc' for i in range(5)], 'average_auc'])
df_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for param in param_list:
    models = []
    ctr = 0
    auc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            # fixed
            is_unbalance=True,
            seed=SEED,
            extra_trees=True,
            min_data_per_group=1,
            early_stopping_round=50,
            # tweak,
            **{
                param_key[0]:param[0],
                param_key[1]:param[1],
                param_key[2]:param[2],
                param_key[3]:param[3],
                'lambda_l1':param[4][0],
                'lambda_l2':param[4][0],
            }
        )
        model.fit(
            X_train, y_train,
            categorical_feature=cat_feature_idx,
            eval_set=(X_val, y_val),
            eval_metric='auc',
            verbose=-1
        )

        y_val_pred = model.predict(X_val)
        auc_score = roc_auc_score(y_val, y_val_pred)

        models.append(model)
        auc_scores.append(auc_score)
        
    df_model.loc[ df_model.shape[0] ] = [
        *param,
        *models,
        *auc_scores,
        sum(auc_scores) / len(auc_scores)
    ]
df_model = df_model.sort_values(by=['average_auc', 'boosting_type', 'learning_rate', 'n_estimators'], ascending=[False, True, True, True]).reset_index(drop=True)
df_model.loc[:1000].to_pickle('model.pkl')
!ls -lah
pd.set_option('display.max_row', df_model.shape[0])
df_model
pd.set_option('display.max_row', 10)
def predict(X, mode='best_mean'):
    if mode == 'best_mean':
        y_preds = []
        for i in range(5):
            y_preds.append(df_model.loc[0, f'model_{i}'].predict(X))
        y_preds = np.mean(np.array(y_preds), axis=0)
    elif mode == 'ensemble_mean':
        y_preds = []
        for i in df_model.index:
            for j in range(5):
                y_preds.append(df_model.loc[i, f'model_{j}'].predict(X))
        y_preds = np.mean(np.array(y_preds), axis=0)
    elif mode == 'weighted_ensemble_mean':
        y_preds = []
#         model_weight = df_model['average_mcc'].apply(lambda a: a/df_model['average_mcc'].sum())
        model_weight = []
        for i in df_model.index:
            model_weight.append(1 + np.log10(df_model.shape[0] - i + 1))
        print(model_weight[:10])
        for i in df_model.index:
            for j in range(5):
                y_preds.append(
                    df_model.loc[i, f'model_{j}'].predict(X) *
                    model_weight[i]
                )
        y_preds = np.array(y_preds)
        y_preds = np.mean(y_preds, axis=0)
    else:
        raise ValueError("Mode isn't supported")
    
    return y_preds
y_test_pred = predict(df_test, mode='best_mean')

df_submission = pd.concat([image_name_test, pd.Series(y_test_pred, name='target')], axis=1)
df_submission.to_csv('submission_best_mean.csv', index=False)

df_submission
y_test_pred2 = predict(df_test, mode='ensemble_mean')

df_submission2 = pd.concat([image_name_test, pd.Series(y_test_pred2, name='target')], axis=1)
df_submission2.to_csv('submission_ensemble_mean.csv', index=False)

df_submission2
y_test_pred3 = predict(df_test, mode='weighted_ensemble_mean')

df_submission3 = pd.concat([image_name_test, pd.Series(y_test_pred3, name='target')], axis=1)
df_submission3.to_csv('submission_weighted_ensemble_mean.csv', index=False)

df_submission3
lgb.plot_importance(df_model.loc[0, 'model_0'], ignore_zero=False, figsize=(16,9))
lgb.plot_tree(df_model.loc[0, 'model_0'], figsize=(32,18))