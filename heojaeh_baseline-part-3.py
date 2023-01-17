# preprocessing

import numpy as np

import pandas as pd 



# graph

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



# model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb



# evaluation

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold, KFold

from bayes_opt import BayesianOptimization



# utils

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
seed = 223
train_act = pd.read_csv('/kaggle/input/train_activity.csv')

train_pay = pd.read_csv('/kaggle/input/train_payment.csv')

train_pledge = pd.read_csv('/kaggle/input/train_pledge.csv')

train_trade = pd.read_csv('/kaggle/input/train_trade.csv')

train_combat = pd.read_csv('/kaggle/input/train_combat.csv')

print('train activity shape: ',train_act.shape)

print('train payment shape: ',train_pay.shape)

print('train pledge shape: ',train_pledge.shape)

print('train trade shape: ',train_trade.shape)

print('train combat shape: ',train_combat.shape)
test1_act = pd.read_csv('/kaggle/input/test1_activity.csv')

test1_pay = pd.read_csv('/kaggle/input/test1_payment.csv')

test1_pledge = pd.read_csv('/kaggle/input/test1_pledge.csv')

test1_trade = pd.read_csv('/kaggle/input/test1_trade.csv')

test1_combat = pd.read_csv('/kaggle/input/test1_combat.csv')

print('test1 activity shape: ',test1_act.shape)

print('test1 payment shape: ',test1_pay.shape)

print('test1 pledge shape: ',test1_pledge.shape)

print('test1 trade shape: ',test1_trade.shape)

print('test1 combat shape: ',test1_combat.shape)
test2_act = pd.read_csv('/kaggle/input/test2_activity.csv')

test2_pay = pd.read_csv('/kaggle/input/test2_payment.csv')

test2_pledge = pd.read_csv('/kaggle/input/test2_pledge.csv')

test2_trade = pd.read_csv('/kaggle/input/test2_trade.csv')

test2_combat = pd.read_csv('/kaggle/input/test2_combat.csv')

print('test2 activity shape: ',test2_act.shape)

print('test2 payment shape: ',test2_pay.shape)

print('test2 pledge shape: ',test2_pledge.shape)

print('test2 trade shape: ',test2_trade.shape)

print('test2 combat shape: ',test2_combat.shape)
train_label = pd.read_csv('/kaggle/input/train_label.csv')

print('train_label shape: ',train_label.shape)
def add_feature_ma3_diff(feature, data, x_df, name):

    # groupby acc_id, day and sum

    f_df = data.groupby(['acc_id','day'])[feature].sum().reset_index()

    

    # make moving average 3 and calculate difference ma3 and feature values

    f_df['ma3'] = f_df.groupby('acc_id')[feature].rolling(3, min_periods=1).mean().values

    f_df['{}_diff'.format(feature)] = f_df['ma3'] - f_df[feature]

    

    # extract last day values

    max_idx = f_df.groupby('acc_id').day.idxmax()

    f_df = f_df.iloc[max_idx, :][['acc_id','{}_diff'.format(feature)]]

    

    # left join on acc_id

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left')

    print('add {} ma3 diff - {} shape: {}'.format(feature, name, x_df.shape))

    

    return x_df
def add_feature_cnt(feature, data, x_df, feature_name, name):

    '''

    add_feature_cnt : 이거는 무슨 함수

    '''

    f_df = data[['acc_id',feature]].drop_duplicates().groupby('acc_id').count().reset_index()

    f_df = f_df.rename(columns={feature:'{}_cnt'.format(feature_name)})

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left')

    print('add {} count - {} shape: {}'.format(feature_name, name, x_df.shape))

    return x_df
def add_day_total_cnt(data, x_df, feature_name, name):

    f_df = data.groupby(['acc_id','server','char_id']).day.count().reset_index().groupby('acc_id').day.sum().reset_index()

    f_df = f_df.rename(columns={'day':'{}_total_cnt'.format(feature_name)})

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left')

    print('add {} total count - {} shape: {}'.format(feature_name, name, x_df.shape))

    return x_df
def add_char_id_total_cnt(data, x_df, feature_name, name):

    f_df = data[['acc_id','server','char_id']].drop_duplicates().groupby('acc_id').char_id.count().reset_index()

    f_df = f_df.rename(columns={'char_id':'{}_total_cnt'.format(feature_name)})

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left')

    print('add {} total count - {} shape: {}'.format(feature_name, name, x_df.shape))

    return x_df
def add_trade_categorical_cnt(feature, user_type, x_df, name):

    f_df = train_trade[[user_type,feature]].pivot_table(index=[user_type,feature], aggfunc='size').reset_index()

    f_df = f_df.set_index([user_type,feature]).unstack()

    # change names

    f_df.columns = ['{}_{}'.format(user_type.split('_')[0], c[1]) for c in f_df.columns]

    f_df = f_df.reset_index().rename(columns={user_type:'acc_id'})

    

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left').fillna(0)

    print("add {}'s {} - {} shape: {}".format(user_type, feature, name, x_df.shape))

    

    return x_df
def add_trade_day_cnt_diff(user_type, x_df, name):

    type_name = user_type.split('_')[0]

    f_df = train_trade[[user_type,'day']].pivot_table(index=[user_type,'day'], aggfunc='size').reset_index()

    trd_total = f_df.groupby(user_type)[0].sum().reset_index().rename(columns={0:'{}_trd_cnt'.format(type_name)})

    trd_7 = f_df[f_df.day>21].groupby(user_type)[0].sum().reset_index().rename(columns={0:'{}_trd7_cnt'.format(type_name)})



    # calc trd count and trd diff

    f_df = pd.merge(trd_total, trd_7, on=user_type, how='left').fillna(0)

    f_df['{}_trd_diff'.format(type_name)] = f_df['{}_trd_cnt'.format(type_name)] - f_df['{}_trd7_cnt'.format(type_name)]

    f_df = f_df.rename(columns={user_type:'acc_id'})

    f_df = f_df[['acc_id','{}_trd_cnt'.format(type_name),'{}_trd_diff'.format(type_name)]]

    

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left').fillna(0)

    print("add {}'s day - {} shape: {}".format(user_type, name, x_df.shape))

        

    return x_df
def add_trade_feature_sum_diff(feature, user_type, x_df, name):

    type_name = user_type.split('_')[0]

    f_df = train_trade.groupby([user_type,'day'])[feature].sum().reset_index()

    f_total = f_df.groupby(user_type)[feature].sum().reset_index().rename(columns={feature:'{}_{}_sum'.format(type_name, feature)})

    f_7 = f_df[f_df.day>21].groupby(user_type)[feature].sum().reset_index().rename(columns={feature:'{}_{}_sum7'.format(type_name, feature)})



    # calc f count and f diff

    f_df = pd.merge(f_total, f_7, on=user_type, how='left').fillna(0)

    f_df['{}_{}_diff'.format(type_name, feature)] = f_df['{}_{}_sum'.format(type_name, feature)] - f_df['{}_{}_sum7'.format(type_name, feature)]

    f_df = f_df.rename(columns={user_type:'acc_id'})

    f_df = f_df[['acc_id','{}_{}_sum'.format(type_name, feature),'{}_{}_diff'.format(type_name, feature)]]

    

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left').fillna(0)

    print("add {}'s {} sum and diff 7 - {} shape: {}".format(user_type, feature, name, x_df.shape))

        

    return x_df
def add_feature_daily_calc(feature, data, x_df, calc, name):

    f_df = data.groupby(['acc_id','day'])[feature].sum().reset_index()

    if calc == 'mean':

        f_df = f_df.groupby('acc_id')[feature].mean().reset_index()

        f_df = f_df.rename(columns={feature:'{}_mean'.format(feature)})

    elif calc == 'median':

        f_df = f_df.groupby('acc_id')[feature].median().reset_index()

        f_df = f_df.rename(columns={feature:'{}_median'.format(feature)})

    elif calc == 'std':

        f_df = f_df.groupby('acc_id')[feature].std().reset_index()

        f_df = f_df.rename(columns={feature:'{}_std'.format(feature)})

    elif calc == 'max':

        f_df = f_df.groupby('acc_id')[feature].max().reset_index()

        f_df = f_df.rename(columns={feature:'{}_max'.format(feature)})

    elif calc == 'min':

        f_df = f_df.groupby('acc_id')[feature].min().reset_index()

        f_df = f_df.rename(columns={feature:'{}_min'.format(feature)})

        

    x_df = pd.merge(x_df, f_df, on='acc_id', how='left')

    print("add {} {} - {} shape: {}".format(feature, name, calc, x_df.shape))

    

    return x_df
train = train_act[['acc_id']].drop_duplicates().copy()

test1 = test1_act[['acc_id']].drop_duplicates().copy()

test2 = test2_act[['acc_id']].drop_duplicates().copy()

print('train shape: ',train.shape)

print('test1 shape: ',test1.shape)

print('test2 shape: ',test2.shape)
# train

train = add_feature_cnt('day', train_act, train, 'act_day', 'train')

train = add_feature_cnt('day', train_combat, train, 'combat_day', 'train')

# test1

test1 = add_feature_cnt('day', test1_act, test1, 'act_day', 'test1')

test1 = add_feature_cnt('day', test1_combat, test1, 'combat_day', 'test1')

# test2

test2 = add_feature_cnt('day', test2_act, test2, 'act_day', 'test2')

test2 = add_feature_cnt('day', test2_combat, test2, 'combat_day', 'test2')
# train

train = add_day_total_cnt(train_act, train, 'act_day', 'train')

train = add_day_total_cnt(train_combat, train, 'combat_day', 'train')

# test1

test1 = add_day_total_cnt(test1_act, test1, 'act_day', 'test1')

test1 = add_day_total_cnt(test1_combat, test1, 'combat_day', 'test1')

# test2

test2 = add_day_total_cnt(test2_act, test2, 'act_day', 'test2')

test2 = add_day_total_cnt(test2_combat, test2, 'combat_day', 'test2')
# train

train = add_char_id_total_cnt(train_act, train, 'act_char_id', 'train')

train = add_char_id_total_cnt(train_combat, train, 'combat_char_id', 'train')

# test1

test1 = add_char_id_total_cnt(test1_act, test1, 'act_char_id', 'test1')

test1 = add_char_id_total_cnt(test1_combat, test1, 'combat_char_id', 'test1')

# test2

test2 = add_char_id_total_cnt(test2_act, test2, 'act_char_id', 'test2')

test2 = add_char_id_total_cnt(test2_combat, test2, 'combat_char_id', 'test2')
act_features = ['playtime','npc_kill','solo_exp','party_exp','quest_exp','rich_monster','death','exp_recovery','game_money_change','fishing','private_shop','enchant_count']

combat_features = ['level','pledge_cnt','random_defender_cnt','temp_cnt','same_pledge_cnt','etc_cnt','num_opponent']

for f in act_features:

    train = add_feature_ma3_diff(f, train_act, train, 'train')

    test1 = add_feature_ma3_diff(f, test1_act, test1, 'test1')

    test2 = add_feature_ma3_diff(f, test2_act, test2, 'test2')

print()

for f in combat_features:

    train = add_feature_ma3_diff(f, train_combat, train, 'train')

    test1 = add_feature_ma3_diff(f, test1_combat, test1, 'test1')

    test2 = add_feature_ma3_diff(f, test2_combat, test2, 'test2')
for f in act_features:

    train = add_feature_daily_calc(f, train_act, train, 'mean', 'train')

    test1 = add_feature_daily_calc(f, test1_act, test1, 'mean', 'test1')

    test2 = add_feature_daily_calc(f, test2_act, test2, 'mean', 'test2')

print()

for f in combat_features:

    train = add_feature_daily_calc(f, train_combat, train, 'mean', 'train')

    test1 = add_feature_daily_calc(f, test1_combat, test1, 'mean', 'test1')

    test2 = add_feature_daily_calc(f, test2_combat, test2, 'mean', 'test2')
train = add_trade_categorical_cnt('item_type','source_acc_id', train, 'train')

test1 = add_trade_categorical_cnt('item_type','source_acc_id', test1, 'test1')

test2 = add_trade_categorical_cnt('item_type','source_acc_id', test2, 'test2')



train = add_trade_categorical_cnt('item_type','target_acc_id', train, 'train')

test1 = add_trade_categorical_cnt('item_type','target_acc_id', test1, 'test1')

test2 = add_trade_categorical_cnt('item_type','target_acc_id', test2, 'test2')
train = add_trade_day_cnt_diff('source_acc_id', train, 'train')

test1 = add_trade_day_cnt_diff('source_acc_id', test1, 'test1')

test2 = add_trade_day_cnt_diff('source_acc_id', test2, 'test2')



train = add_trade_day_cnt_diff('target_acc_id', train, 'train')

test1 = add_trade_day_cnt_diff('target_acc_id', test1, 'test1')

test2 = add_trade_day_cnt_diff('target_acc_id', test2, 'test2')
for f in ['item_amount','type']:

    train = add_trade_feature_sum_diff(f, 'source_acc_id', train, 'train')

    test1 = add_trade_feature_sum_diff(f, 'source_acc_id', test1, 'test1')

    test2 = add_trade_feature_sum_diff(f, 'source_acc_id', test2, 'test2')



    train = add_trade_feature_sum_diff(f, 'target_acc_id', train, 'train')

    test1 = add_trade_feature_sum_diff(f, 'target_acc_id', test1, 'test1')

    test2 = add_trade_feature_sum_diff(f, 'target_acc_id', test2, 'test2')
pledge_cnt_sum_df = train_combat.groupby('acc_id').agg({'pledge_cnt':'sum','num_opponent':'sum'})

pledge_cnt_sum_df.columns = ['total_pledge_cnt','total_num_opponent']

train = pd.merge(train, pledge_cnt_sum_df, on='acc_id', how='left')

print('train 크기 체크: ',train.shape)



pledge_cnt_sum_df = test1_combat.groupby('acc_id').agg({'pledge_cnt':'sum','num_opponent':'sum'})

pledge_cnt_sum_df.columns = ['total_pledge_cnt','total_num_opponent']

test1 = pd.merge(test1, pledge_cnt_sum_df, on='acc_id', how='left')

print('test1 크기 체크: ',test1.shape)



pledge_cnt_sum_df = test2_combat.groupby('acc_id').agg({'pledge_cnt':'sum','num_opponent':'sum'})

pledge_cnt_sum_df.columns = ['total_pledge_cnt','total_num_opponent']

test2 = pd.merge(test2, pledge_cnt_sum_df, on='acc_id', how='left')

print('test2 크기 체크: ',test2.shape)
total_pledge_id_df = train_pledge[['acc_id','char_id','server','pledge_id']].drop_duplicates().groupby('acc_id').pledge_id.count().reset_index()

total_pledge_id_df.columns = ['acc_id','total_pledge_id']

train = pd.merge(train, total_pledge_id_df, on='acc_id', how='left')

train = train.fillna(0)

print('train 크기 체크: ',train.shape)



total_pledge_id_df = test1_pledge[['acc_id','char_id','server','pledge_id']].drop_duplicates().groupby('acc_id').pledge_id.count().reset_index()

total_pledge_id_df.columns = ['acc_id','total_pledge_id']

test1 = pd.merge(test1, total_pledge_id_df, on='acc_id', how='left')

test1 = test1.fillna(0)

print('test1 크기 체크: ',test1.shape)



total_pledge_id_df = test2_pledge[['acc_id','char_id','server','pledge_id']].drop_duplicates().groupby('acc_id').pledge_id.count().reset_index()

total_pledge_id_df.columns = ['acc_id','total_pledge_id']

test2 = pd.merge(test2, total_pledge_id_df, on='acc_id', how='left')

test2 = test2.fillna(0)

print('test2 크기 체크: ',test2.shape)
amount_by_acc_id = train_pay.groupby('acc_id').agg({'amount_spent':['max','median']})

amount_by_acc_id.columns = ['max_amount','median_amount']

train = pd.merge(train, amount_by_acc_id, on='acc_id', how='left')

train = train.fillna(0)

print('train 크기 체크: ',train.shape)
amount_by_acc_id = test1_pay.groupby('acc_id').agg({'amount_spent':['max','median']})

amount_by_acc_id.columns = ['max_amount','median_amount']

test1 = pd.merge(test1, amount_by_acc_id, on='acc_id', how='left')

test1 = test1.fillna(0)

print('test1 크기 체크: ',test1.shape)
amount_by_acc_id = test2_pay.groupby('acc_id').agg({'amount_spent':['max','median']})

amount_by_acc_id.columns = ['max_amount','median_amount']

test2 = pd.merge(test2, amount_by_acc_id, on='acc_id', how='left')

test2 = test2.fillna(0)

print('test2 크기 체크: ',test2.shape)
amount_features = ['etc_cnt_diff',

                 'random_defender_cnt_mean',

                 'npc_kill_mean',

                 'solo_exp_mean',

                 'playtime_mean',

                 'game_money_change_mean',

                 'source_enchant_scroll',

                 'num_opponent_diff',

                 'level_diff',

                 'playtime_diff',

                 'fishing_diff',

                 'npc_kill_diff',

                 'quest_exp_diff',

                 'game_money_change_diff',

                 'temp_cnt_mean',

                 'source_item_amount_diff',

                 'etc_cnt_mean',

                 'source_item_amount_sum',

                 'solo_exp_diff',

                 'death_diff',

                 'party_exp_diff',

                 'max_amount']
survival_features = ['playtime_mean',

                     'playtime_diff',

                     'npc_kill_mean',

                     'npc_kill_diff',

                     'game_money_change_mean',

                     'quest_exp_mean',

                     'solo_exp_mean',

                     'solo_exp_diff',

                     'level_mean',

                     'quest_exp_diff',

                     'game_money_change_diff',

                     'fishing_mean',

                     'temp_cnt_mean',

                     'death_mean',

                     'etc_cnt_mean',

                     'target_item_amount_diff',

                     'target_item_amount_sum',

                     'num_opponent_mean',

                     'source_item_amount_sum',

                     'fishing_diff',

                     'median_amount',

                     'party_exp_mean']
def survival64(y_pred, dataset):

    y_true = dataset.get_label()

    y_pred = np.array([64 if x > 64 else x for x in y_pred])

    y_pred = np.array([0 if x < 0 else x for x in y_pred])

    y_pred = np.round(y_pred)

    error = metrics.mean_absolute_error(y_true, y_pred)

    return 'error', error, False
def amount_postprocessing(pred):

    return pd.Series(pred).apply(lambda x: 0 if x < 0 else x)



def survival_postprocessing(pred):

    pred = pd.Series(pred).apply(lambda x: 64 if x > 64 else x)

    return pred.apply(lambda x: 1 if x < 0 else x).round()
kf = KFold(n_splits=5, random_state=seed)
x_data = train.set_index('acc_id')

y_amount_train = train_label.amount_spent

y_survival_train = train_label.survival_time

y_survival_train_week = y_survival_train // 7



x_test1 = test1.set_index('acc_id')

x_test2 = test2.set_index('acc_id')
def LGB_amount_evaluate(**params):

    params['n_estimators'] = int(round(params['n_estimators'],0))

    params['eta'] = int(round(params['eta'],0))

    params['num_leaves'] = int(round(params['num_leaves'],0))

    params['colsample_bytree'] = int(round(params['colsample_bytree'],0))

        

    test_pred = pd.DataFrame()

    

    for n_fold, (train_idx, valid_idx) in enumerate(kf.split(x_data, y_amount_train)):

        X_train, X_valid = x_data.iloc[train_idx], x_data.iloc[valid_idx]

        y_train, y_valid = y_amount_train[train_idx], y_amount_train[valid_idx]

        

        model = lgb.LGBMRegressor(**params,

                                  random_state=seed,

                                  verbose=0,

                                  metric='mae')

        model.fit(X_train, y_train)



        y_pred_valid = model.predict(X_valid)

        y_pred_valid = amount_postprocessing(y_pred_valid)

    

        test_pred['{}_fold'.format(n_fold)] = y_pred_valid

        

    # 음수를 붙인 이유는 Target이 높은값이어야하기 때문이다.

    return -metrics.mean_absolute_error(y_valid, test_pred.mean(axis=1))
def LGB_survival_evaluate(**params):

    params['n_estimators'] = int(round(params['n_estimators'],0))

    params['eta'] = int(round(params['eta'],4))

    params['num_leaves'] = int(round(params['num_leaves'],0))

    params['colsample_bytree'] = int(round(params['colsample_bytree'],0))

        

    test_pred = pd.DataFrame()

    

    for n_fold, (train_idx, valid_idx) in enumerate(kf.split(x_data, y_survival_train)):

        X_train, X_valid = x_data.iloc[train_idx], x_data.iloc[valid_idx]

        y_train, y_valid = y_survival_train[train_idx], y_survival_train[valid_idx]

        

        model = lgb.LGBMRegressor(**params,

                                  random_state=seed,

                                  verbose=0,

                                  metric='mae')

        model.fit(X_train, y_train)



        y_pred_valid = model.predict(X_valid)

        y_pred_valid = survival_postprocessing(y_pred_valid)

    

        test_pred['{}_fold'.format(n_fold)] = y_pred_valid

        

    # 음수를 붙인 이유는 Target이 높은값이어야하기 때문이다.

    return -metrics.mean_absolute_error(y_valid, test_pred.mean(axis=1))
lgb_param_grid = {

    'n_estimators':(500,2000),

    'eta':(0.01, 0.5),

    'num_leaves':(30,50),

    'bagging_fraction':(0.5,1),

    'feature_fraction':(0.5,1),

    'colsample_bytree':(0.5,1)

}



lgb_week_param_grid = {

    'n_estimators':(500,2000),

    'eta':(0.01, 0.5),

    'num_leaves':(30,50),

    'bagging_fraction':(0.5,1),

    'feature_fraction':(0.5,1),

    'colsample_bytree':(0.5,1)

}
lgb_b_o_amount = BayesianOptimization(LGB_amount_evaluate, lgb_param_grid, random_state=seed)

lgb_b_o_amount.maximize(init_points=5, n_iter=20)
lgb_b_o_survival = BayesianOptimization(LGB_survival_evaluate, lgb_param_grid, random_state=seed)

lgb_b_o_survival.maximize(init_points=5, n_iter=20)
lgb_amount_bp = dict()

for k, v in lgb_b_o_amount.max['params'].items():

    lgb_amount_bp[k] = v



lgb_survival_bp = dict()

for k, v in lgb_b_o_survival.max['params'].items():

    lgb_survival_bp[k] = v



# num_leaves

lgb_amount_bp['num_leaves'] = int(np.round(lgb_amount_bp['num_leaves']))

lgb_survival_bp['num_leaves'] = int(np.round(lgb_survival_bp['num_leaves']))

# n_estimators

lgb_amount_bp['n_estimators'] = int(np.round(lgb_amount_bp['n_estimators']))

lgb_survival_bp['n_estimators'] = int(np.round(lgb_survival_bp['n_estimators']))
lgb_amount_bp = {'bagging_fraction': 0.8927746477285299,

                 'colsample_bytree': 0.9606149102852795,

                 'eta': 0.4164215305825649,

                 'feature_fraction': 0.840013600879963,

                 'n_estimators': 500,

                 'num_leaves': 30}
lgb_survival_bp = {'bagging_fraction': 0.5,

                     'colsample_bytree': 0.5,

                     'eta': 0.01,

                     'feature_fraction': 0.5,

                     'n_estimators': 653,

                     'num_leaves': 50}
# Amount LGB Model

lgb_amount_model = lgb.LGBMRegressor(**lgb_amount_bp,

                                     random_state=seed,

                                     verbose=0,

                                     metric='mae')

lgb_amount_model.fit(x_data[amount_features], y_amount_train)

# light GBM amount spent

lgb_amount_pred_test1 = lgb_amount_model.predict(x_test1[amount_features])

lgb_amount_pred_test2 = lgb_amount_model.predict(x_test2[amount_features])





# Survival LGB Model

lgb_survival_model = lgb.LGBMRegressor(**lgb_survival_bp,

                                       random_state=seed,

                                       verbose=0,

                                       metric='mae')

lgb_survival_model.fit(x_data[survival_features], y_survival_train)

# light GBM survival time

lgb_survival_pred_test1 = lgb_survival_model.predict(x_test1[survival_features])

lgb_survival_pred_test2 = lgb_survival_model.predict(x_test2[survival_features])



# Post Processing

lgb_amount_pred_test1 = amount_postprocessing(lgb_amount_pred_test1)

lgb_amount_pred_test2 = amount_postprocessing(lgb_amount_pred_test2)

lgb_survival_pred_test1 = survival_postprocessing(lgb_survival_pred_test1)

lgb_survival_pred_test2 = survival_postprocessing(lgb_survival_pred_test2)



# concat

test1_lgb = pd.DataFrame({'acc_id':test1.acc_id,

                         'survival_time':lgb_survival_pred_test1,

                         'amount_spent':lgb_amount_pred_test1})

test2_lgb = pd.DataFrame({'acc_id':test2.acc_id,

                         'survival_time':lgb_survival_pred_test2,

                         'amount_spent':lgb_amount_pred_test2})

# print

print('Light GBM Test1 Prediction shape: ',test1_lgb.shape)

print('Light GBM Test2 Prediction shape: ',test2_lgb.shape)
test1_lgb.to_csv('test1_predict.csv', index=False)

test2_lgb.to_csv('test2_predict.csv', index=False)
f_score_amount = pd.DataFrame({'feature':x_data[amount_features].columns,

                               'score':lgb_amount_model.feature_importances_})
f_score_survival = pd.DataFrame({'feature':x_data[survival_features].columns,

                               'score':lgb_survival_model.feature_importances_})
f, ax = plt.subplots(1,2,figsize=(15,5))

sns.barplot(x='score', y='feature', data=f_score_amount.sort_values(by='score',ascending=False), ax=ax[0])

sns.barplot(x='score', y='feature', data=f_score_survival.sort_values(by='score',ascending=False), ax=ax[1])

ax[0].set_title('Amount Spent', size=15)

ax[1].set_title('Survival Time', size=15)

plt.show()
def score_function(predict, actual):

    

    # predict = pd.read_csv(predict_label, engine='python') # 예측 답안 파일 불러오기

    # actual = pd.read_csv(actual_label,engine='python') # 실제 답안 파일 불러오기



    predict.acc_id = predict.acc_id.astype('int')

    predict = predict.sort_values(by =['acc_id'], axis = 0) # 예측 답안을 acc_id 기준으로 정렬 

    predict = predict.reset_index(drop = True)

    actual.acc_id = actual.acc_id.astype('int')

    actual = actual.sort_values(by =['acc_id'], axis = 0) # 실제 답안을 acc_id 기준으로 정렬

    actual =actual.reset_index(drop=True)

    

    if predict.acc_id.equals(actual.acc_id) == False:

        print('acc_id of predicted and actual label does not match')

        sys.exit() # 예측 답안의 acc_id와 실제 답안의 acc_id가 다른 경우 에러처리 

    else:

            

        S, alpha, L, sigma = 30, 0.01, 0.1, 15  

        cost, gamma, add_rev = 0,0,0 

        profit_result = []

        survival_time_pred = list(predict.survival_time)

        amount_spent_pred = list(predict.amount_spent)

        survival_time_actual = list(actual.survival_time)

        amount_spent_actual = list(actual.amount_spent)    

        for i in range(len(survival_time_pred)):

            if survival_time_pred[i] == 64 :                 

                cost = 0

                optimal_cost = 0

            else:

                cost = alpha * S * amount_spent_pred[i]                    #비용 계산

                optimal_cost = alpha * S * amount_spent_actual[i]          #적정비용 계산 

            

            if optimal_cost == 0:

                gamma = 0

            elif cost / optimal_cost < L:

                gamma = 0

            elif cost / optimal_cost >= 1:

                gamma = 1

            else:

                gamma = (cost)/((1-L)*optimal_cost) - L/(1-L)              #반응률 계산

            

            if survival_time_pred[i] == 64 or survival_time_actual[i] == 64:

                T_k = 0

            else:

                T_k = S * np.exp(-((survival_time_pred[i] - survival_time_actual[i])**2)/(2*(sigma)**2))    #추가 생존기간 계산

                

            add_rev = T_k * amount_spent_actual[i]                         #잔존가치 계산

    

           

            profit = gamma * add_rev - cost                                #유저별 기대이익 계산

            profit_result.append(profit)

            

        score = sum(profit_result) 

    return score
y_amount_train_pred = lgb_amount_model.predict(x_data[amount_features])

y_survival_train_pred = lgb_survival_model.predict(x_data[survival_features])



y_amount_train_pred = amount_postprocessing(y_amount_train_pred)

y_survival_train_pred = survival_postprocessing(y_survival_train_pred)
train_pred_df = pd.DataFrame({'acc_id':train.acc_id,

                              'survival_time':y_survival_train_pred,

                              'amount_spent':y_amount_train_pred})
train_score = score_function(train_pred_df, train_label)

true_score = score_function(train_label,train_label)
print('Train score: ',train_score)

print('true score: ',true_score)
pd.merge(train_pred_df, train_label, on='acc_id', how='left').head(20)