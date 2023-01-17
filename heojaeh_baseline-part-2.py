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

from sklearn.metrics import mean_squared_error



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
validation_acc = pd.read_csv('/kaggle/input/train_valid_user_id.csv')

print('validation_acc shape: ',validation_acc.shape)
train = train_act.groupby(['acc_id','server','char_id']).day.count().reset_index().groupby('acc_id').day.sum().reset_index()

train.columns = ['acc_id','total_days']

train = pd.merge(train, train_label, on='acc_id', how='inner')

print('train 크기 체크: ',train.shape)
test1 = test1_act.groupby(['acc_id','server','char_id']).day.count().reset_index().groupby('acc_id').day.sum().reset_index()

test1.columns = ['acc_id','total_days']

print('test1 크기 체크: ',test1.shape)
test2 = test2_act.groupby(['acc_id','server','char_id']).day.count().reset_index().groupby('acc_id').day.sum().reset_index()

test2.columns = ['acc_id','total_days']

print('test2 크기 체크: ',test2.shape)
total_char_id_df = train_act[['acc_id','server','char_id']].drop_duplicates().groupby('acc_id').char_id.count().reset_index()

total_char_id_df.columns = ['acc_id','total_char_id']

train = pd.merge(train, total_char_id_df, on='acc_id', how='left')

print('train 크기 체크: ',train.shape)
total_char_id_df = test1_act[['acc_id','server','char_id']].drop_duplicates().groupby('acc_id').char_id.count().reset_index()

total_char_id_df.columns = ['acc_id','total_char_id']

test1 = pd.merge(test1, total_char_id_df, on='acc_id', how='left')

print('test1 크기 체크: ',test1.shape)
total_char_id_df = test2_act[['acc_id','server','char_id']].drop_duplicates().groupby('acc_id').char_id.count().reset_index()

total_char_id_df.columns = ['acc_id','total_char_id']

test2 = pd.merge(test2, total_char_id_df, on='acc_id', how='left')

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
playtime_by_day = train_act.groupby(['day','acc_id']).playtime.sum().reset_index().groupby('acc_id').agg({'playtime':['std','mean']})

playtime_by_day.columns = ['std_playtime','avg_playtime']

train = pd.merge(train, playtime_by_day, on='acc_id', how='left')

train = train.fillna(0)

print('train 크기 체크: ',train.shape)
playtime_by_day = test1_act.groupby(['day','acc_id']).playtime.sum().reset_index().groupby('acc_id').agg({'playtime':['std','mean']})

playtime_by_day.columns = ['std_playtime','avg_playtime']

test1 = pd.merge(test1, playtime_by_day, on='acc_id', how='left')

test1 = test1.fillna(0)

print('test1 크기 체크: ',test1.shape)
playtime_by_day = test2_act.groupby(['day','acc_id']).playtime.sum().reset_index().groupby('acc_id').agg({'playtime':['std','mean']})

playtime_by_day.columns = ['std_playtime','avg_playtime']

test2 = pd.merge(test2, playtime_by_day, on='acc_id', how='left')

test2 = test2.fillna(0)

print('test2 크기 체크: ',test2.shape)
sales_cnt_df = train_trade.source_acc_id.value_counts().reset_index()

sales_cnt_df.columns = ['acc_id','sales_cnt']

train = pd.merge(train, sales_cnt_df, on='acc_id', how='left')

train = train.fillna(0)

print('train 크기 체크: ',train.shape)
sales_cnt_df = test1_trade.source_acc_id.value_counts().reset_index()

sales_cnt_df.columns = ['acc_id','sales_cnt']

test1 = pd.merge(test1, sales_cnt_df, on='acc_id', how='left')

test1 = test1.fillna(0)

print('test1 크기 체크: ',test1.shape)
sales_cnt_df = test2_trade.source_acc_id.value_counts().reset_index()

sales_cnt_df.columns = ['acc_id','sales_cnt']

test2 = pd.merge(test2, sales_cnt_df, on='acc_id', how='left')

test2 = test2.fillna(0)

print('test2 크기 체크: ',test2.shape)
total_amount_df = train_trade[train_trade.source_acc_id.isin(train.acc_id)].groupby('source_acc_id').item_amount.sum().reset_index()

total_amount_df.columns = ['acc_id','total_amount']

train = pd.merge(train, total_amount_df, on='acc_id', how='left')

train = train.fillna(0)

print('train 크기 체크: ',train.shape)
total_amount_df = test1_trade[test1_trade.source_acc_id.isin(test1.acc_id)].groupby('source_acc_id').item_amount.sum().reset_index()

total_amount_df.columns = ['acc_id','total_amount']

test1 = pd.merge(test1, total_amount_df, on='acc_id', how='left')

test1 = test1.fillna(0)

print('test1 크기 체크: ',test1.shape)
total_amount_df = test2_trade[test2_trade.source_acc_id.isin(test2.acc_id)].groupby('source_acc_id').item_amount.sum().reset_index()

total_amount_df.columns = ['acc_id','total_amount']

test2 = pd.merge(test2, total_amount_df, on='acc_id', how='left')

test2 = test2.fillna(0)

print('test2 크기 체크: ',test2.shape)
class_by_user = train_combat[['acc_id','server','char_id','class']].drop_duplicates().groupby(['acc_id','class']).char_id.count().unstack()

class_by_user = class_by_user.fillna(0)

class_by_user.columns = ['class{}'.format(i) for i in range(8)]

train = pd.merge(train, class_by_user, on='acc_id', how='left')

print('train 크기 체크: ',train.shape)
class_by_user = test1_combat[['acc_id','server','char_id','class']].drop_duplicates().groupby(['acc_id','class']).char_id.count().unstack()

class_by_user = class_by_user.fillna(0)

class_by_user.columns = ['class{}'.format(i) for i in range(8)]

test1 = pd.merge(test1, class_by_user, on='acc_id', how='left')

print('test1 크기 체크: ',test1.shape)
class_by_user = test2_combat[['acc_id','server','char_id','class']].drop_duplicates().groupby(['acc_id','class']).char_id.count().unstack()

class_by_user = class_by_user.fillna(0)

class_by_user.columns = ['class{}'.format(i) for i in range(8)]

test2 = pd.merge(test2, class_by_user, on='acc_id', how='left')

print('test2 크기 체크: ',test2.shape)
random_df = train_combat.groupby('acc_id').agg({'random_attacker_cnt':'sum','random_defender_cnt':'sum'})

random_df.columns = ['total_random_attacker_cnt','total_random_defender_cnt']

train = pd.merge(train, random_df, on='acc_id', how='left')

print('train 크기 체크: ',train.shape)
random_df = test1_combat.groupby('acc_id').agg({'random_attacker_cnt':'sum','random_defender_cnt':'sum'})

random_df.columns = ['total_random_attacker_cnt','total_random_defender_cnt']

test1 = pd.merge(test1, random_df, on='acc_id', how='left')

print('test1 크기 체크: ',test1.shape)
random_df = test2_combat.groupby('acc_id').agg({'random_attacker_cnt':'sum','random_defender_cnt':'sum'})

random_df.columns = ['total_random_attacker_cnt','total_random_defender_cnt']

test2 = pd.merge(test2, random_df, on='acc_id', how='left')

print('test2 크기 체크: ',test2.shape)
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
f, ax = plt.subplots(1,5,figsize=(30,5))

sns.scatterplot(train.total_pledge_id, train.survival_time,ax=ax[0])

ax[0].set_title('Correlation Total pledge ID and Survival Time')

ax[0].set_xlabel('Total pledge ID')

ax[0].set_ylabel('Survival Time')

ax[0].axvline(20, color='red', linestyle='--')

sns.scatterplot(train.total_num_opponent, train.survival_time,ax=ax[1])

ax[1].set_title('Correlation Total Number of Opponent and Survival Time')

ax[1].set_xlabel('Total Number of Opponent')

ax[1].set_ylabel('Survival Time')

ax[1].axvline(100, color='red', linestyle='--')

sns.scatterplot(train.total_pledge_cnt, train.survival_time,ax=ax[2])

ax[2].set_title('Correlation Total pledge Count and Survival Time')

ax[2].set_xlabel('Total pledge Count')

ax[2].set_ylabel('Survival Time')

ax[2].axvline(100, color='red', linestyle='--')

sns.scatterplot(train.sales_cnt, train.survival_time,ax=ax[3])

ax[3].set_title('Correlation Sales Count and Survival Time')

ax[3].set_xlabel('Sales Count')

ax[3].set_ylabel('Survival Time')

ax[3].axvline(1000, color='red', linestyle='--')

sns.scatterplot(train.avg_playtime, train.survival_time,ax=ax[4])

ax[4].set_title('Correlation Average Playtime and Survival Time', color='red')

ax[4].set_xlabel('Average Playtime')

ax[4].set_ylabel('Survival Time')

plt.show()



train64 = train[train.survival_time==64]



f, ax = plt.subplots(1,4,figsize=(24,3))

sns.distplot(train64.total_pledge_id, ax=ax[0])

ax[0].set_title('Histogram of Total pledge ID with Non-Churn')

ax[0].set_xlabel('Total pledge ID')

ax[0].axvline(20, color='red', linestyle='--')

sns.distplot(train64.total_num_opponent, ax=ax[1])

ax[1].set_title('Histogram of Total Number of Opponent with Non-Churn')

ax[1].set_xlabel('Total Number of Opponent')

ax[1].axvline(100, color='red', linestyle='--')

sns.distplot(train64.total_pledge_cnt, ax=ax[2])

ax[2].set_title('Histogram of Total pledge Count with Non-Churn')

ax[2].set_xlabel('Total pledge Count')

ax[2].axvline(100, color='red', linestyle='--')

sns.distplot(train64.sales_cnt, ax=ax[3])

ax[3].set_title('Histogram of Sales Count with Non-Churn')

ax[3].set_xlabel('Sales Count')

ax[3].axvline(1000, color='red', linestyle='--')

plt.show()
f, ax = plt.subplots(1,2,figsize=(15,5))

sns.scatterplot(train.total_random_defender_cnt, train.survival_time, ax=ax[0])

ax[0].set_title('Correlation Random Defender Count and Survival Time')

ax[0].set_xlabel('Random Defender Count')

ax[0].set_ylabel('Survival Time')

ax[0].axvline(100, color='red', linestyle='--')

sns.scatterplot(train.total_random_attacker_cnt, train.survival_time, ax=ax[1])

ax[1].set_title('Correlation Random Attacker Count and Survival Time')

ax[1].set_xlabel('Random Attacker Count')

ax[1].set_ylabel('Survival Time')

ax[1].axvline(200, color='red', linestyle='--')

plt.show()



train64 = train[train.survival_time==64]



f, ax = plt.subplots(1,2,figsize=(15,3))

sns.distplot(train64.total_pledge_id, ax=ax[0])

ax[0].set_title('Histogram of Random Defender Count with Non-Churn')

ax[0].set_xlabel('Random Defender Count')

ax[0].axvline(100, color='red', linestyle='--')

sns.distplot(train64.total_num_opponent, ax=ax[1])

ax[1].set_title('Histogram of Random Attacker Count with Non-Churn')

ax[1].set_xlabel('Random Attacker Count')

ax[1].axvline(200, color='red', linestyle='--')

plt.show()
_, ax = plt.subplots(2,4, figsize=(20,10))

f = ['class0','class1','class2','class3','class4','class5','class6','class7']

l = [20,50,19,10,10,10,40,20]

for i in range(8):

    sns.scatterplot(train[f[i]], train.survival_time, ax=ax[i//4,i%4])

    ax[i//4,i%4].set_ylabel('Survival Time')

    ax[i//4,i%4].axvline(l[i], color='red', linestyle='--')

    if i != 6:

        ax[i//4,i%4].set_title('Correlation {} and Survival Time'.format(f[i]))

    else:

        ax[i//4,i%4].set_title('Correlation {} and Survival Time'.format(f[i]), color='red')
train_idx = validation_acc[validation_acc.set=='Train'].acc_id

valid_idx = validation_acc[validation_acc.set=='Validation'].acc_id



train_set = train[train.acc_id.isin(train_idx)]

valid_set = train[train.acc_id.isin(valid_idx)]



print('train set: ',train_set.shape)

print('valid set: ',valid_set.shape)
def survival64(y_pred, dataset):

    y_true = dataset.get_label()

    y_pred = np.array([64 if x > 64 else x for x in y_pred])

    y_pred = np.array([0 if x < 0 else x for x in y_pred])

    y_pred = np.round(y_pred)

    error = np.sqrt(mean_squared_error(y_true, y_pred))

    return 'error', error, False
lr_amount = LinearRegression()

lr_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

          train_set.amount_spent)

lr_amount_pred = lr_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lr_amount_pred = pd.Series(lr_amount_pred).apply(lambda x: 0 if x < 0 else x)
lr_survival = LinearRegression()

lr_survival.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

          train_set.survival_time)

lr_survival_pred = lr_survival.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lr_survival_pred = pd.Series(lr_survival_pred).apply(lambda x: 64 if x > 64 else x)

lr_survival_pred = lr_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
lr_pred_df = pd.DataFrame({'acc_id':valid_set.acc_id.values,

                           'survival_time':lr_survival_pred,

                           'amount_spent':lr_amount_pred})

print('lr_pred_df shape: ',lr_pred_df.shape)
# amount

lr_amount_pred_test1 = lr_amount.predict(test1.drop(['acc_id'], axis=1))

lr_amount_pred_test2 = lr_amount.predict(test2.drop(['acc_id'], axis=1))

# survival time

lr_survival_pred_test1 = lr_survival.predict(test1.drop(['acc_id'], axis=1))

lr_survival_pred_test2 = lr_survival.predict(test2.drop(['acc_id'], axis=1))

# concat

test1_lr = pd.DataFrame({'acc_id':test1.acc_id,

                         'survival_time':lr_survival_pred_test1,

                         'amount_spent':lr_amount_pred_test1})

test2_lr = pd.DataFrame({'acc_id':test2.acc_id,

                         'survival_time':lr_survival_pred_test2,

                         'amount_spent':lr_amount_pred_test2})

# print

print('Linear Regression Test1 Prediction shape: ',test1_lr.shape)

print('Linear Regression Test2 Prediction shape: ',test2_lr.shape)
rf_params = {

    'n_estimators':1000,

    'max_depth':10,

    'n_jobs':5,

    'random_state':seed

}
rf_amount = RandomForestRegressor(**rf_params)

rf_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

              train_set.amount_spent)

rf_amount_pred = rf_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_amount_pred = pd.Series(rf_amount_pred).apply(lambda x: 0 if x < 0 else x)
rf_survival = RandomForestRegressor(**rf_params)

rf_survival.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                train_set.survival_time)

rf_survival_pred = rf_survival.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_survival_pred = pd.Series(rf_survival_pred).apply(lambda x: 64 if x > 64 else x)

rf_survival_pred = rf_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
rf_pred_df = pd.DataFrame({'acc_id':valid_set.acc_id.values,

                           'survival_time':rf_survival_pred,

                           'amount_spent':rf_amount_pred})

print('rf_pred_df shape: ',rf_pred_df.shape)
# amount

rf_amount_pred_test1 = rf_amount.predict(test1.drop(['acc_id'], axis=1))

rf_amount_pred_test2 = rf_amount.predict(test2.drop(['acc_id'], axis=1))

# survival time

rf_survival_pred_test1 = rf_survival.predict(test1.drop(['acc_id'], axis=1))

rf_survival_pred_test2 = rf_survival.predict(test2.drop(['acc_id'], axis=1))

# concat

test1_rf = pd.DataFrame({'acc_id':test1.acc_id,

                         'survival_time':rf_survival_pred_test1,

                         'amount_spent':rf_amount_pred_test1})

test2_rf = pd.DataFrame({'acc_id':test2.acc_id,

                         'survival_time':rf_survival_pred_test2,

                         'amount_spent':rf_amount_pred_test2})

# print

print('Random Forest Test1 Prediction shape: ',test1_lr.shape)

print('Random Forest Test2 Prediction shape: ',test2_lr.shape)
lgb_params = {

    'n_estimators':800,

    'sees':seed

}
lgb_train_amount = lgb.Dataset(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                               train_set.amount_spent)

lgb_train_survival = lgb.Dataset(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                                 train_set.survival_time)
lgb_amount = lgb.train(lgb_params, 

                       lgb_train_amount,

                       feval=survival64,

                       valid_sets=[lgb_train_amount],

                       verbose_eval=100)



lgb_amount_pred = lgb_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lgb_amount_pred = pd.Series(lgb_amount_pred).apply(lambda x: 0 if x < 0 else x)
lgb_survival = lgb.train(lgb_params, 

                         lgb_train_survival,

                         feval=survival64,

                         valid_sets=[lgb_train_survival],

                         verbose_eval=100)



lgb_survival_pred = lgb_survival.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lgb_survival_pred = pd.Series(lgb_survival_pred).apply(lambda x: 64 if x > 64 else x)

lgb_survival_pred = lgb_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
lgb_pred_df = pd.DataFrame({'acc_id':valid_set.acc_id.values,

                           'survival_time':lgb_survival_pred,

                           'amount_spent':lgb_amount_pred})

print('lgb_pred_df shape: ',lgb_pred_df.shape)
# amount

lgb_amount_pred_test1 = lgb_amount.predict(test1.drop(['acc_id'], axis=1))

lgb_amount_pred_test2 = lgb_amount.predict(test2.drop(['acc_id'], axis=1))

# survival time

lgb_survival_pred_test1 = lgb_survival.predict(test1.drop(['acc_id'], axis=1))

lgb_survival_pred_test2 = lgb_survival.predict(test2.drop(['acc_id'], axis=1))

# concat

test1_lgb = pd.DataFrame({'acc_id':test1.acc_id,

                         'survival_time':lgb_survival_pred_test1,

                         'amount_spent':lgb_amount_pred_test1})

test2_lgb = pd.DataFrame({'acc_id':test2.acc_id,

                         'survival_time':lgb_survival_pred_test2,

                         'amount_spent':lgb_amount_pred_test2})

# print

print('Light GBM Test1 Prediction shape: ',test1_lr.shape)

print('Light GBM Test2 Prediction shape: ',test2_lr.shape)
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
lr_valid_score = score_function(lr_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

rf_valid_score = score_function(rf_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

lgb_valid_score = score_function(lgb_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

true_score = score_function(valid_set[['acc_id','survival_time','amount_spent']],

                            valid_set[['acc_id','survival_time','amount_spent']])



print('Linear Regression score: ',lr_valid_score)

print('Random Forest score: ',rf_valid_score)

print('Light GBM score: ',lgb_valid_score)

print('true score: ',true_score)
# Linear Regression

test1_lr.to_csv('test1_lr_tootouch_baseline2.csv', index=False)

test2_lr.to_csv('test2_lr_tootouch_baseline2.csv', index=False)

# Randaom Forest

test1_rf.to_csv('test1_rf_tootouch_baseline2.csv', index=False)

test2_rf.to_csv('test2_rf_tootouch_baseline2.csv', index=False)

# Light GBM

test1_lgb.to_csv('test1_lgb_tootouch_baseline2.csv', index=False)

test2_lgb.to_csv('test2_lgb_tootouch_baseline2.csv', index=False)