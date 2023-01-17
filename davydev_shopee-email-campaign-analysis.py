import os

import random

import platform

import itertools



import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.width', None)

pd.set_option('display.max_column', None)



SEED = 42



os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)



print('Python version:', platform.python_version())

print('Numpy version:', np.__version__)

print('Pandas version:', pd.__version__)

print('Matplotlib version:', matplotlib.__version__)

print('Seaborn version:', sns.__version__)
basedir = '../input/shopee-code-league-20/_DA_Marketing_Analytics'



# converting to dataframe

df_train = pd.read_csv(os.path.join(basedir,'train.csv'))

df_test = pd.read_csv(os.path.join(basedir,'test.csv'))

df_users = pd.read_csv(os.path.join(basedir,'users.csv'))
df_train.dtypes
df_test.dtypes
df_users.dtypes
df_train.isnull().sum()
df_test.isnull().sum()
df_users.isnull().sum()
print(np.sort(df_train['country_code'].unique()))

print(np.sort(df_test['country_code'].unique()))
print(np.sort(df_users['attr_1'].unique()))

print(np.sort(df_users['attr_2'].unique()))

print(np.sort(df_users['attr_3'].unique()))

print(np.sort(df_users['domain'].unique()))
df_train.corr()
list_unique = df_users['domain'].unique()

dict_unique = {list_unique[i]: i for i in range(len(list_unique))}

df_users['domain'] = df_users['domain'].apply(lambda d: dict_unique[d])
def convert(day):

    try:

        return np.float(day)

    except:

        return np.nan
df_train['last_open_day'] = df_train['last_open_day'].apply(convert)

df_train['last_login_day'] = df_train['last_login_day'].apply(convert)

df_train['last_checkout_day'] = df_train['last_checkout_day'].apply(convert)



df_test['last_open_day'] = df_test['last_open_day'].apply(convert)

df_test['last_login_day'] = df_test['last_login_day'].apply(convert)

df_test['last_checkout_day'] = df_test['last_checkout_day'].apply(convert)
df_train = df_train.join(df_users, on='user_id', rsuffix='_unused')

df_test = df_test.join(df_users, on='user_id', rsuffix='_unused')
del df_train['user_id']

del df_train['user_id_unused']

del df_train['row_id']



del df_test['user_id']

del df_test['user_id_unused']

del df_test['row_id']
df_train['day'] = pd.to_datetime(df_train['grass_date']).dt.dayofweek.astype('category')

df_test['day'] = pd.to_datetime(df_test['grass_date']).dt.dayofweek.astype('category')
del df_train['grass_date']

del df_test['grass_date']
def fix_age(age):

    if age < 18 or age >= 100:

        return np.nan

    else:

        return age

    

df_train['age'] = df_train['age'].apply(fix_age)

df_test['age'] = df_test['age'].apply(fix_age)
# domain

# 1 -> 'other' domain from previous preprocessing

# df_train['domain_nan'] = df_train['domain'].isnull()

df_train['domain'] = df_train['domain'].fillna(1)



# df_test['domain_nan'] = df_test['domain'].isnull()

df_test['domain'] = df_test['domain'].fillna(1)
df_train
df_test
df_users
df_train.to_csv('train_processed.csv', index=False)

df_test.to_csv('test_processed.csv', index=False)

df_users.to_csv('users_processed.csv', index=False)
df_train.to_parquet('train_processed.parquet', engine='pyarrow')

df_test.to_parquet('test_processed.parquet', engine='pyarrow')

df_users.to_parquet('users_processed.parquet', engine='pyarrow')
import sklearn

import lightgbm as lgbm

import scipy



print('Scikit-Learn version:', sklearn.__version__)

print('LightGBM version:', lgbm.__version__)

print('Scipy version:', scipy.__version__)
X = df_train.copy()

del X['open_flag']



X_test = df_test.copy()



y = df_train['open_flag'].to_numpy()
cat_feature = [

    'country_code','attr_1', 'attr_2', 'attr_3',

    'domain','day',

#     'last_open_day_nan', 'last_login_day_nan',

#     'last_checkout_day_nan', 'attr_1_nan', 'attr_2_nan',

#     'attr_3_nan', 'age_nan', 'domain_nan',

    

]

cat_feature_idx = [X.columns.get_loc(ct) for ct in cat_feature]

cat_feature_idx
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import matthews_corrcoef



K = [3, 5, 10]
param_dict = {

    'learning_rate': [0.0075, 0.01, 0.0125],

    'min_data_in_leaf': [20, 50],

    'max_bin': [16, 102, 255],

    'lambda': [

        # l1, l2

        [0.0, 0.0],

        [0.001, 0.01],

        [0.01, 0.1],

        [1.0, 0.01],

    ],

    'n_estimators': [100, 125, 150]

}

param_key = list(param_dict.keys())

param_item = list(param_dict.values())

param_item
param_list = list(itertools.product(*param_item))

param_list[:10]
len(param_list)
df_model = pd.DataFrame(columns=[*param_key, *[f'model_{i}' for i in range(sum(K))], *[f'model_{i}_mcc' for i in range(sum(K))], 'average_mcc'])

df_model
skf_list = [StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED) for k in K]



for param in param_list:

    ctr = 0

    model = []

    mcc_score = []

    for skf in skf_list:

        for train_idx, val_idx in skf.split(X, y):

            X_train, X_val = X.loc[train_idx], X.loc[val_idx]

            y_train, y_val = y[train_idx], y[val_idx]



            model.append(

                lgbm.LGBMClassifier(

                    # fixed

                    is_unbalance=True,

                    seed=SEED,

                    extra_trees=True,



                    min_data_per_group=1,

                    boosting_type='goss',

                    num_leaves=63,

                    feature_fraction=0.9,

                    # variable

                    learning_rate=param[0],

                    min_data_in_leaf=param[1],

                    max_bin=param[2], 

                    lambda_l1=param[3][0],

                    lambda_l2=param[3][1],

                    n_estimators=param[4],

                )

            )

            model[ctr].fit(

                X_train, y_train,

                categorical_feature=cat_feature_idx

            )



            y_val_pred = model[ctr].predict(X_val)

            mcc_score.append(matthews_corrcoef(y_val, y_val_pred))



            ctr += 1

    df_model.loc[ df_model.shape[0] ] = [

        *param,

        *model,

        *mcc_score,

        sum(mcc_score) / len(mcc_score)

    ]
df_model = df_model.sort_values(by=['average_mcc', 'learning_rate'], ascending=[False, True]).reset_index(drop=True)

df_model.loc[:1000].to_pickle('model.pkl')

!ls -lah
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, matthews_corrcoef



def predict(X, mode='best_mean'):

    if mode == 'best_mode':

        y_preds = []

        for i in range(sum(K)):

            y_preds.append(df_model.loc[0, f'model_{i}'].predict(X))

        y_preds = np.array(y_preds)

        y_preds = scipy.stats.mode(y_preds)

        y_preds = y_preds[0]

        y_preds = y_preds.reshape(-1)

    elif mode == 'best_mean':

        y_preds = []

        for i in range(sum(K)):

            y_preds.append(df_model.loc[0, f'model_{i}'].predict_proba(X))

        y_preds = np.mean(np.array(y_preds), axis=0)

        y_preds = np.argmax(y_preds, axis=-1)

    elif mode == 'ensemble_mode':

        y_preds = []

        for i in df_model.index:

            for j in range(sum(K)):

                y_preds.append(df_model.loc[i, f'model_{j}'].predict(X))

        y_preds = np.array(y_preds)

        y_preds = scipy.stats.mode(y_preds)

        y_preds = y_preds[0]

        y_preds = y_preds.reshape(-1)

    elif mode == 'ensemble_mean':

        y_preds = []

        for i in df_model.index:

            for j in range(sum(K)):

                y_preds.append(df_model.loc[i, f'model_{j}'].predict_proba(X))

        y_preds = np.mean(np.array(y_preds), axis=0)

        y_preds = np.argmax(y_preds, axis=-1)

    elif mode == 'weighted_ensemble_mean':

        y_preds = []

#         model_weight = df_model['average_mcc'].apply(lambda a: a/df_model['average_mcc'].sum())

        model_weight = []

        for i in df_model.index:

            model_weight.append(1 + np.log10(df_model.shape[0] - i + 1))

        print(model_weight[:10])

        for i in df_model.index:

            for j in range(sum(K)):

                y_preds.append(

                    df_model.loc[i, f'model_{j}'].predict_proba(X) *

                    model_weight[i]

                )

        y_preds = np.array(y_preds)

        y_preds = np.mean(y_preds, axis=0)

        y_preds = np.argmax(y_preds, axis=-1)

    else:

        raise ValueError("Mode isn't supported")

    

    return y_preds



def metrics(y_true, y_pred):

    print('Weighted F1 Score :', f1_score(y_true, y_pred, average='weighted'))

    print('MCC Score :', matthews_corrcoef(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    cm = pd.DataFrame(cm, [0, 1], [0, 1])



    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")

    plt.show()
y_train_pred = predict(X_train, mode='best_mode')

metrics(y_train, y_train_pred)
y_train_pred2 = predict(X_train, mode='best_mean')

metrics(y_train, y_train_pred2)
y_train_pred3 = predict(X_train, mode='ensemble_mode')

metrics(y_train, y_train_pred3)
y_train_pred4 = predict(X_train, mode='ensemble_mean')

metrics(y_train, y_train_pred4)
y_train_pred5 = predict(X_train, mode='weighted_ensemble_mean')

metrics(y_train, y_train_pred5)
pred_modes = ['best_mode','best_mean','ensemble_mode','ensemble_mean','weighted_ensemble_mean']



for mdx in pred_modes:

    y_test_pred = predict(X_test, mode=mdx)

    df_submission = pd.concat([pd.Series(list(range(0, len(X_test))), name='row_id', dtype=np.int32), pd.Series(y_test_pred, name='open_flag')], axis=1)

    df_submission.to_csv('submission_'+mdx+'.csv', index=False)
lgbm.plot_importance(df_model.loc[0, 'model_0'], ignore_zero=False, figsize=(16,9))
lgbm.plot_split_value_histogram(df_model.loc[0, 'model_0'], 2)