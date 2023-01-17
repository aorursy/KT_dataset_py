import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 12)

import math

import seaborn as sns

sns.set(style='darkgrid')

import matplotlib.pyplot as plt



from catboost import CatBoostRegressor

import lightgbm as lgb

from sklearn.model_selection import KFold



import warnings 

warnings.filterwarnings('ignore')
PATH = '../input'

train_df = pd.read_csv(f'{PATH}/train.csv', low_memory=False, parse_dates=['date'], index_col=['date'])

test_df = pd.read_csv(f'{PATH}/test.csv', low_memory=False, parse_dates=['date'], index_col=['date'])

sample_submission = pd.read_csv(f'{PATH}/sample_submission.csv')



split_idx = train_df.shape[0]

full_df = pd.concat([train_df, test_df], sort=False)



# del train_df, test_df
def create_future_main(df):

    '''Creating main features for a date'''

    print('Run future_main')

    

    df['year'] = df.index.year

    df['half_year'] = (df.index.month - 1) // 6

    df['quarter'] = df.index.quarter

    df['month'] = df.index.month

    df['week'] = df.index.week

    

    df['day_of_year'] = df.index.dayofyear

    df['day_of_month'] = df.index.day

    df['day_of_week'] = df.index.dayofweek

    

    seasons = [1,1,2,2,2,3,3,3,4,4,4,1]

    df['season'] = df['month'].apply(lambda month: seasons[month-1])

    

    df['year_plus_half_year'] = df.index.year * 10 + ((df.index.month - 1) // 6)

    df['year_plus_quarter'] = df.index.year * 10 + df.index.quarter

    df['year_plus_month'] = df.index.year * 100 + df.index.month

    df['year_plus_week'] = df.index.year * 100 + df.index.week

    

    df['total_days_from_start'] = (df.index - df.index.min()).days

    

    # Sine-cosine representation of a date/time

    df['day_of_year_sin'] = df['day_of_year'].apply(lambda day: np.sin(2 * math.pi * day/24.))

    df['day_of_year_cos'] = df['day_of_year'].apply(lambda day: np.cos(2 * math.pi * day/24.))

    df['day_of_month_sin'] = df['day_of_month'].apply(lambda month: np.sin(2 * math.pi * month/24.))

    df['day_of_month_cos'] = df['day_of_month'].apply(lambda month: np.cos(2 * math.pi * month/24.))

    df['day_of_week_sin'] = df['day_of_week'].apply(lambda week: np.sin(2 * math.pi * week/24.))

    df['day_of_week_cos'] = df['day_of_week'].apply(lambda week: np.cos(2 * math.pi * week/24.))

    

    return df



def create_future_group(df):

    '''Create aggregate futures for date/sales. The create_future_main function must be executed first.'''

    print('Run future_group')

    

    aggr_func = ['mean','median','sum','min','max','std']



    print(' - group1')

    for gr in ['store','item']:

        for aggr in aggr_func:

            df[f'group1_{gr}_{aggr}'] = df.groupby(gr)['sales'].transform(aggr)



    print(' - group2')

    for gr1 in ['store','item']:

        for gr2 in ['half_year','quarter','month','week','day_of_year','day_of_month','day_of_week','season']:

            for aggr in aggr_func:

                df[f'group2_{gr1}_{gr2}_{aggr}'] = df.groupby([gr1,gr2])['sales'].transform(aggr)



    print(' - group3')

    for gr in ['half_year','quarter','month','week','day_of_year','day_of_month','day_of_week','season']:

        for aggr in aggr_func:

            df[f'group3_store_item_{gr}_{aggr}'] = df.groupby(['store','item',gr])['sales'].transform(aggr)



    print(' - group4')

    dbl_group = [

        ['half_year','month'],

        ['quarter','month'],

        ['month','day_of_month'],

        ['month','day_of_week'],

        ['week','day_of_week'],

        ['season','day_of_month'],

        ['season','day_of_week']

    ]

    

    for gr in dbl_group:

            for aggr in aggr_func:

                df[f'group4_store_item_{gr1[0]}_{gr[1]}_{aggr}'] = df.groupby(['store','item',gr[0],gr[1]])['sales'].transform(aggr)

    

    return df



def create_future_shift(df):

    '''Creating aggregate futures by date/sales for past periods. The create_future_main function must be executed first.'''

    print('Run future_shift')

    

    # Day lag

    for lag in [1,2,3,4,5,10,20,30,60,90,120,150,180,210,240,270,300,330,365]:

        full_df[f'shift_day_store_item_{lag}'] = full_df.groupby(['store','item'])['sales'].transform(lambda x: x.shift(lag))

        full_df[f'shift_day_store_item_{lag}'].fillna(full_df[f'shift_day_store_item_{lag}'].mode()[0], inplace=True)

    

    # Week lag with aggregation

    for lag in [1,2,3,4,8,12,52]:

        for aggr in ['mean','median','sum','min','max','std']:

            full_df[f'shift_week_store_item_{lag}_{aggr}'] = full_df.groupby(['store','item','week'])['sales'].transform(

                lambda x: x.shift(12).aggregate(aggr))

            full_df[f'shift_week_store_item_{lag}_{aggr}'].fillna(full_df[f'shift_week_store_item_{lag}_{aggr}'].mode()[0], inplace=True)

    

    return df



def create_future_synthetic(df):

    '''Creating synthetic future for the date. The create_future_main function must be executed first.'''

    print('Run future_synthetic')

    

    for i in range(1,13):

        df[f'div_month_{i}'] = df['Month'] // i

    

    for i in range(1,53):

        df[f'div_week_{i}'] = df['week'] // i

    

    for i in range(1,357):

        df[f'div_day_of_year_{i}'] = df['day_of_year'] // i

    

    for i in range(1,32):

        df[f'div_day_of_month_{i}'] = df['day_of_month'] // i

        

    for i in range(1,8):

        df[f'div_day_of_week_{i}'] = df['day_of_week'] // i

    

    return df

    

def create_future_binary(df):

    '''Creating binary futures for the date. The create_future_main function must be executed first.'''

    print('Run future_binary')

    

    df['day_of_week_by'] = df['day_of_week']

    df = pd.get_dummies(df, columns = ['day_of_week_by']) # quarter, month, week, day_of_year, day_of_month, season

    

    df['is_weekend'] =  df['day_of_week'].apply(lambda day: day > 5).astype(int)

    

    df['is_month_start'] = df.index.is_month_start.astype(int)

    df['is_month_end'] = df.index.is_month_end.astype(int)



    return df



def create_future_last(df,split_idx):

    '''Creation of aggregating futures for the last N days. The create_future_main function must be executed first.'''

    print('Run future_last')

       

    maxt = df.iloc[[split_idx-1]].index.values[0]



    for l in [7,14,30,60,90,180,360]:

        mint = maxt - np.timedelta64(l,'D')

        for aggr in ['mean','median','sum','min','max','std']:

            for gr in ['store','item']:

                temt_df = df.query('index >= @mint and index <= @maxt').groupby(gr)['sales'].aggregate(aggr)

                temt_df.rename(columns={'sales':f'last1_{gr}_{aggr}_{l}'}, inplace=True)

                df = df.merge(pd.DataFrame(temt_df), left_on=gr, right_index=True)

                

            temt_df = df.query('index >= @mint and index <= @maxt').groupby(['store','item'])['sales'].aggregate(aggr)

            temt_df.rename(columns={'sales':f'last2_store_item_{aggr}_{l}'}, inplace=True)

            df = df.merge(pd.DataFrame(temt_df), left_on=['store','item'], right_index=True)

  

    return df
%%time

full_df = create_future_main(full_df)

full_df = create_future_group(full_df)

full_df = create_future_shift(full_df)

#full_df = create_future_binary(full_df)

#full_df = create_future_last(full_df,split_idx)
def lgbm_smape(preds, train_data):

    smape_val = smape(np.expm1(preds), np.expm1(train_data))

    return 'SMAPE', smape_val, False



def smape(preds, target):

    n = len(preds)

    masked_arr = ~((preds==0)&(target==0))

    preds, target = preds[masked_arr], target[masked_arr]

    num = np.abs(preds-target)

    denom = np.abs(preds)+np.abs(target)

    smape_val = (20*np.sum(num/denom))/n

    return smape_val



def kaggle_smape(true, predicted):

    true_o = true

    pred_o = predicted

    summ = np.abs(true_o) + np.abs(pred_o)

    smape = np.where(summ==0, 0, np.abs(pred_o - true_o) / summ)

    return smape
model_cols = [c for c in full_df.columns if c not in ['sales','id']]



X = full_df[:split_idx][model_cols]

y = full_df[:split_idx]['sales']

X_test = full_df[split_idx:][model_cols]



oof_preds = np.zeros([X.shape[0]])

sub_preds = np.zeros([X_test.shape[0]])



folds = KFold(n_splits=3, shuffle=True, random_state=42)



cat_cols = ['store','item','half_year','quarter','month','week','day_of_year','day_of_month','day_of_week','season']
params_lgb = {

    'nthread': -1,

    'categorical_feature': [X.columns.get_loc(c) for c in cat_cols],

    'task': 'train',

    'max_depth': 6, # 8

    'num_leaves': 10, # 127

    'boosting_type': 'gbdt',

    'objective': 'regression', # regression_l1

    'metric': 'mae',    

    'learning_rate': 0.2, # 0.25

    #'feature_fraction': 0.9,

    #'bagging_fraction': 0.8,

    #'bagging_freq': 30,

    #'lambda_l1': 0.06,

    #'lambda_l2': 0.1,

    'verbose': -1

}



feature_importance_df = pd.DataFrame()
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):

    

    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]

    valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]

       

    lgb_model = lgb.LGBMRegressor(**params_lgb,n_estimators=3000, n_jobs=-1)

    lgb_model.fit(

        train_x,

        train_y,

        eval_set = [(train_x, train_y), (valid_x, valid_y)],

        verbose = 200, 

        early_stopping_rounds = 200,

        eval_metric = lgbm_smape

    )

    

    oof_preds[valid_idx] = lgb_model.predict(valid_x, num_iteration = lgb_model.best_iteration_)

    sub_preds[:] += lgb_model.predict(X_test, num_iteration = lgb_model.best_iteration_) / folds.n_splits

    

  

    importance_df = pd.DataFrame()

    importance_df['feature'] = X.columns

    importance_df['importance'] = lgb_model.feature_importances_

    importance_df['fold'] = n_fold + 1

    

    feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
feature_importance_df[['feature','importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)[:12]
sample_submission['sales'] = sub_preds.astype(int)

sample_submission.to_csv('submission_lgb.csv', index=False)
params_cat = {

    'iterations': 1000,

    'max_ctr_complexity': 6,

    'random_seed': 42,

    'od_type': 'Iter',

    'od_wait': 50,

    'verbose': 50,

    'depth': 6

}



feature_importance_df = pd.DataFrame()
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):

    

    train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]

    valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]

        

    cat_model = CatBoostRegressor(eval_metric='MAE', **params_cat)

    cat_model.fit(

        train_x, 

        train_y, 

        eval_set=(valid_x, valid_y),

        cat_features=[X.columns.get_loc(c) for c in cat_cols],

        use_best_model=True

    )

    

    oof_preds[valid_idx] = cat_model.predict(valid_x)

    sub_preds[:] += cat_model.predict(X_test) / folds.n_splits

    

    print('SMAPE validation: {}'.format(lgbm_smape(valid_x, valid_y)))

    

    importance_df = pd.DataFrame()

    importance_df['feature'] = X.columns

    importance_df['importance'] = cat_model.feature_importances_

    importance_df['fold'] = n_fold + 1

    

    feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
feature_importance_df[['feature','importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)[:12]
sample_submission['sales'] = sub_preds.astype(int)

sample_submission.to_csv('submission_cat.csv', index=False)
full_df
full_df.describe()
full_df.info()
g = sns.FacetGrid(full_df, row='store', col='month', margin_titles=True)

g.map(plt.hist, 'sales', color='steelblue', bins=np.linspace(0, 200, 11))
g = sns.FacetGrid(full_df, col='season', margin_titles=True)

g.map(plt.hist, 'sales', color='steelblue', bins=np.linspace(0, 200, 11))
g = sns.FacetGrid(full_df, col='day_of_week', margin_titles=True)

g.map(plt.hist, 'sales', color='steelblue', bins=np.linspace(0, 200, 11))
g = sns.FacetGrid(full_df, col='store', margin_titles=True)

g.map(plt.hist, 'sales', color='steelblue', bins=np.linspace(0, 200, 11))
g = sns.FacetGrid(full_df, col='day_of_week', margin_titles=True)

g.map(plt.hist, 'sales', color='steelblue', bins=np.linspace(0, 200, 11))
f, axes = plt.subplots(1, 3, figsize=(25, 5))

sns.distplot(full_df[:split_idx]['sales'], kde=False, color='b', ax=axes[0])

sns.barplot(x='day_of_week', y='sales', data=full_df, estimator=sum, color='b', ax=axes[1])

sns.barplot(x='season', y='sales', data=full_df, estimator=sum, color='b', ax=axes[2])
f, axes = plt.subplots(figsize=(25, 7))

sns.barplot(x='day_of_week', y='sales', hue='store', estimator=sum, data=full_df, color='b')
f, axes = plt.subplots(figsize=(17, 5))

sns.barplot(x='day_of_month', y='sales', data=full_df, estimator=sum, color='b')
f, axes = plt.subplots(figsize=(20, 5))

sns.barplot(x='week', y='sales', data=full_df, estimator=sum, color='b')
f, axes = plt.subplots(figsize=(20, 10))

sns.lineplot(x='week', y='sales', hue='year', data=full_df, estimator=sum, color='b')