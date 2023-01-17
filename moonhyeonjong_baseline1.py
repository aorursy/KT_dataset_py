import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns



matplotlib.rcParams['axes.unicode_minus'] = False # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처

matplotlib.rcParams["font.family"] = "Malgun Gothic" 



import warnings

warnings.filterwarnings('ignore')
def week(x):

    if x <= 7:

        return 1

    elif x <= 14:

        return 2

    elif x <= 21:

        return 3

    elif x <= 28:

        return 4

    elif x <= 35:

        return 5

    elif x <= 42:

        return 6

    elif x <= 49:

        return 7

    elif x <= 56:

        return 8

    elif x <= 63:

        return 9

    else:

        return 10

    

def make_variable_about_week(df, col, new_df):

    change = df[['acc_id','server','char_id',col,'week']].drop_duplicates().groupby(['acc_id','week'])[col].sum().unstack()

    change = change.fillna(0)

    change.columns = ['{}_week{}'.format(col,i+1) for i in range(4)]

    new_df = pd.merge(df, change, on = 'acc_id', how = 'left')

    

    return new_df



def is_survival(x):

    if x == 64:

        return 1

    else :

        0
train_activity = pd.read_csv('../input/bigcontest2019/train_activity.csv')

train_combat = pd.read_csv('../input/bigcontest2019/train_combat.csv')

train_pledge = pd.read_csv('../input/bigcontest2019/train_pledge.csv')

train_trade = pd.read_csv('../input/bigcontest2019/train_trade.csv')

train_payment = pd.read_csv('../input/bigcontest2019/train_payment.csv')



label = pd.read_csv('../input/bigcontest2019/train_label.csv')
test1_activity = pd.read_csv('../input/bigcontest2019/test1_activity.csv')

test1_combat = pd.read_csv('../input/bigcontest2019/test1_combat.csv')

test1_pledge = pd.read_csv('../input/bigcontest2019/test1_pledge.csv')

test1_trade = pd.read_csv('../input/bigcontest2019/test1_trade.csv')

test1_payment = pd.read_csv('../input/bigcontest2019/test1_payment.csv')
test2_activity = pd.read_csv('../input/bigcontest2019/test2_activity.csv')

test2_combat = pd.read_csv('../input/bigcontest2019/test2_combat.csv')

test2_pledge = pd.read_csv('../input/bigcontest2019/test2_pledge.csv')

test2_trade = pd.read_csv('../input/bigcontest2019/test2_trade.csv')

test2_payment = pd.read_csv('../input/bigcontest2019/test2_payment.csv')
train_activity['data'] = 'train'

test1_activity['data'] = 'test1'

test2_activity['data'] = 'test2'



activity = pd.concat([train_activity, test1_activity, test2_activity], axis = 0)



activity = pd.merge(activity, label, on = 'acc_id', how = 'left')
train_combat['data'] = 'train'

test1_combat['data'] = 'test1'

test2_combat['data'] = 'test2'



combat = pd.concat([train_combat, test1_combat, test2_combat], axis = 0)



combat = pd.merge(combat, label, on = 'acc_id', how = 'left')
train_pledge['data'] = 'train'

test1_pledge['data'] = 'test1'

test2_pledge['data'] = 'test2'



pledge = pd.concat([train_pledge, test1_pledge, test2_pledge], axis = 0)



pledge = pd.merge(pledge, label,  on = 'acc_id', how = 'left')
train_payment['data'] = 'train'

test1_payment['data'] = 'test1'

test2_payment['data'] = 'test2'



payment = pd.concat([train_payment, test1_payment, test2_payment], axis = 0)



payment = pd.merge(payment, label,  on = 'acc_id', how = 'left')
train_trade['data'] = 'train'

test1_trade['data'] = 'test1'

test2_trade['data'] = 'test2'



trade = pd.concat([train_trade, test1_trade, test2_trade], axis = 0)

trade['hour'] = trade['time'].str.split(':', expand = True)[0]



trade_buy = trade.copy()

trade_sell = trade.copy()



trade_buy['trade_type'] = 'buy'

trade_sell['trade_type'] = 'sell'



trade_buy.drop(['source_acc_id', 'source_char_id'], axis = 1, inplace = True)

trade_sell.drop(['target_acc_id', 'target_char_id'], axis = 1, inplace = True)



trade_buy = trade_buy.rename(columns = {'target_acc_id' : 'acc_id' , 'target_char_id' : 'char_id'})

trade_sell = trade_sell.rename(columns = {'source_acc_id' : 'acc_id', 'source_char_id' : 'char_id'})



total_trade = pd.concat([trade_buy, trade_sell], axis = 0)



trade = pd.merge(total_trade, label,on = 'acc_id', how = 'left')
activity['week'] = activity['day'].apply(week)
activity.head()
# game_money_change는 절댓값으로 변환



activity['game_money_change'] = np.abs(activity['game_money_change'])
# 어떤 유형으로 경험치를 얻는지에 대한 비율

activity['total_exp'] = activity['solo_exp'] + activity['party_exp'] + activity['quest_exp']



activity['solo_exp_ratio'] = activity['solo_exp'] / activity['total_exp']

activity['party_exp_ratio'] = activity['party_exp'] / activity['total_exp']

activity['quest_exp_ratio'] = activity['quest_exp'] / activity['total_exp']



activity[['solo_exp_ratio', 'party_exp_ratio', 'quest_exp_ratio']] = activity[['solo_exp_ratio', 'party_exp_ratio', 'quest_exp_ratio']].fillna(0)
activity.loc[(activity['survival_time'].notnull())].groupby(['acc_id', 'week'])['playtime', 'npc_kill', 'solo_exp', 'party_exp', 'quest_exp', 'rich_monster',

                                     'death', 'revive', 'fishing', 'private_shop', 'game_money_change',

                                     'solo_exp_ratio', 'party_exp_ratio', 'quest_exp_ratio', 'survival_time'].mean().round(4)
activity_mean = pd.DataFrame(activity.groupby(['acc_id'])['playtime', 'npc_kill', 'solo_exp', 'party_exp', 'quest_exp', 'rich_monster',

                                                          'death', 'revive', 'fishing', 'private_shop', 'game_money_change',

                                                          'solo_exp_ratio', 'party_exp_ratio', 'quest_exp_ratio'].mean()).reset_index()
activity_mean.head()
df_activity = activity.groupby(['acc_id'])['fishing', 'private_shop', 'game_money_change'].mean()
combat['week'] = combat['day'].apply(week)
combat.head()
combat.columns
# 공격적 성향인지, 막피로 피해를 받았는지에 대한 비율

combat[['acc_id', 'random_attacker_cnt', 'random_defender_cnt']]



combat['random_attacker_ratio'] = combat['random_attacker_cnt'] / (combat['random_attacker_cnt'] + combat['random_defender_cnt'])

combat['random_defender_ratio'] = combat['random_defender_cnt'] / (combat['random_attacker_cnt'] + combat['random_defender_cnt'])



combat[['random_attacker_ratio', 'random_defender_ratio']] = combat[['random_attacker_ratio', 'random_defender_ratio']].fillna(0)
# 어떤전투의 유형의 전투를 했는지 

combat['total_combat'] = combat['pledge_cnt'] + combat['temp_cnt'] + combat['same_pledge_cnt'] + combat['etc_cnt']



combat['pledge_cnt_ratio'] = combat['pledge_cnt'] / combat['total_combat']

combat['temp_cnt_ratio'] = combat['temp_cnt'] / combat['total_combat']

combat['same_pledge_cnt_ratio']  = combat['same_pledge_cnt'] / combat['total_combat']

combat['etc_cnt_ratio'] = combat['etc_cnt'] / combat['total_combat']



combat[['pledge_cnt_ratio','temp_cnt_ratio', 'same_pledge_cnt_ratio', 'etc_cnt_ratio']] = combat[['pledge_cnt_ratio','temp_cnt_ratio', 'same_pledge_cnt_ratio', 'etc_cnt_ratio']].fillna(0)
combat.columns
combat_mean = combat.groupby(['acc_id'])['pledge_cnt', 'random_attacker_cnt', 'random_defender_cnt',

                                         'temp_cnt','same_pledge_cnt', 'etc_cnt', 'num_opponent', 'level_max', 

                                         'random_attacker_ratio', 'random_defender_ratio',

                                         'pledge_cnt_ratio', 'temp_cnt_ratio','same_pledge_cnt_ratio', 'etc_cnt_ratio'].mean().reset_index()
df_combat = combat_mean.copy()
pledge['week'] = pledge['day'].apply(week)
pledge.columns
# 혈맹 중 접속혈맹 캐릭터수에 비해 전투 참여 혈맹 캐릭터 수에 비율이 얼마인지?

# 혈맹이 활발한지 아닌지의 유무를 파악

pledge['play_combat_char_ratio'] = pledge['combat_char_cnt'] / pledge['play_char_cnt']
# 공격적 성향인지, 막피로 피해를 받았는지에 대한 비율

pledge[['acc_id','pledge_id','random_attacker_cnt', 'random_defender_cnt']]



pledge['random_attacker_ratio'] = pledge['random_attacker_cnt'] / (pledge['random_attacker_cnt'] + pledge['random_defender_cnt'])

pledge['random_defender_ratio'] = pledge['random_defender_cnt'] / (pledge['random_attacker_cnt'] + pledge['random_defender_cnt'])



pledge[['random_attacker_ratio', 'random_defender_ratio']] = pledge[['random_attacker_ratio', 'random_defender_ratio']].fillna(0)
# 어떤전투의 유형의 전투를 했는지 

pledge['total_combat'] =  pledge['pledge_combat_cnt'] + pledge['temp_cnt'] + pledge['same_pledge_cnt'] + pledge['etc_cnt']



pledge['pledge_combat_cnt_ratio'] = pledge['pledge_combat_cnt'] / pledge['total_combat']

pledge['temp_cnt_ratio'] = pledge['temp_cnt'] / pledge['total_combat']

pledge['same_pledge_cnt_ratio']  = pledge['same_pledge_cnt'] / pledge['total_combat']

pledge['etc_cnt_ratio'] = pledge['etc_cnt'] / pledge['total_combat']



pledge[['pledge_combat_cnt_ratio', 'temp_cnt_ratio', 'same_pledge_cnt_ratio', 'etc_cnt_ratio']] =pledge[['pledge_combat_cnt_ratio', 'temp_cnt_ratio', 'same_pledge_cnt_ratio', 'etc_cnt_ratio']].fillna(0)
# 혈맹에서 전투 캐릭터인지 비전투 캐릭터인지 

pledge['combat_play_time_ratio'] = pledge['combat_play_time'] / (pledge['combat_play_time'] + pledge['non_combat_play_time'])

pledge['non_combat_play_time_ratio'] = pledge['non_combat_play_time'] / (pledge['combat_play_time'] + pledge['non_combat_play_time'])
pledge.head()
pledge.columns
pledge_mean = pledge.groupby(['acc_id'])['play_char_cnt', 'combat_char_cnt', 'pledge_combat_cnt',

                                         'random_attacker_cnt','random_defender_cnt', 'etc_cnt',

                                         'combat_play_time', 'non_combat_play_time',

                                         'random_attacker_ratio', 'random_defender_ratio',

                                         'pledge_combat_cnt_ratio','etc_cnt_ratio',

                                         'combat_play_time_ratio', 'non_combat_play_time_ratio'].mean().reset_index()
df_pledge = pledge_mean.copy()
trade['week'] = trade['day'].apply(week)
trade.columns
item_type = pd.crosstab(trade['acc_id'], trade['item_type'], margins = True)

item_type.rename(columns = {'All': 'item_type_all'}, inplace = True)

trade_type = pd.crosstab(trade['acc_id'], trade['trade_type'], margins = True)

trade_type.rename(columns = {'All': 'trade_type_all'}, inplace = True)



tp = pd.merge(item_type, trade_type, on = 'acc_id', how = 'left')



trade = pd.merge(trade, tp, on = 'acc_id', how = 'left')
trade_server = trade[['acc_id','server']].drop_duplicates().groupby('acc_id').server.count().reset_index(name='server_cnt')

trade_char = trade[['acc_id','char_id','server']].drop_duplicates().groupby('acc_id').char_id.count().reset_index(name='char_cnt')



df_train = pd.merge(df_train, trade_server, on='acc_id', how='left')

df_train = pd.merge(df_train, trade_char, on='acc_id', how='left')
trade.head()
df_trade = trade.groupby(['acc_id'])['item_amount', 'server_cnt', 'char_id_cnt',

                                     'accessory', 'adena', 'armor','enchant_scroll', 'etc', 'spell', 'weapon', 'item_type_all',

                                     'buy', 'sell', 'trade_type_all'].mean()
payment.head()
payment['week'] = payment['day'].apply(week)



payment.rename(columns = {'amount_spent_x' : 'payment_amount_spent', 'amount_spent_y' : 'amount_spent'}, inplace = True)
df_payment = payment.groupby(['acc_id'])['payment_amount_spent'].mean()
data1 = pd.merge(df_activity, df_combat, on = 'acc_id', how = 'left')

data2 = pd.merge(df_pledge, df_trade, on = 'acc_id', how = 'left')

data = pd.merge(data1, data2, on = 'acc_id', how = 'left')

data = pd.merge(data, df_payment, on = 'acc_id', how = 'left')
data = pd.merge(df_activity, label, on = 'acc_id', how = 'left')
data.head()
data.shape
# model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

import xgboost as xgb



# evaluation

from sklearn.metrics import mean_squared_error
seed = 223
validation_acc = pd.read_csv('../input/bigcontest2019/train_valid_user_id.csv')

print('validation_acc shape: ',validation_acc.shape)
train_idx = validation_acc[validation_acc.set=='Train'].acc_id

valid_idx = validation_acc[validation_acc.set=='Validation'].acc_id



train_set = data[data.acc_id.isin(train_idx)]

valid_set = data[data.acc_id.isin(valid_idx)]



print('train set: ',train_set.shape)

print('valid set: ',valid_set.shape)
train_set.fillna(0, inplace = True)

valid_set.fillna(0, inplace = True)
def survival64(y_pred, dataset):

    y_true = dataset.get_label()

    y_pred = np.array([64 if x > 64 else x for x in y_pred])

    y_pred = np.array([0 if x < 0 else x for x in y_pred])

    y_pred = np.round(y_pred)

    error = np.sqrt(mean_squared_error(y_true, y_pred))

    return 'error', error, False
lr_amount = RandomForestRegressor()

lr_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1), train_set.amount_spent)

lr_amount_pred = lr_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lr_amount_pred = pd.Series(lr_amount_pred).apply(lambda x: 0 if x < 0 else x)
lr_survival = RandomForestRegressor()

lr_survival.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

          train_set.survival_time)

lr_survival_pred = lr_survival.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lr_survival_pred = pd.Series(lr_survival_pred).apply(lambda x: 64 if x > 64 else x)

lr_survival_pred = lr_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
lr_pred_df = pd.DataFrame({'acc_id':valid_set.acc_id.values,

                           'survival_time':lr_survival_pred,

                           'amount_spent':lr_amount_pred})

print('lr_pred_df shape: ',lr_pred_df.shape)
rf_params = {

    'n_estimators':1000,

    'max_depth':10,

    'n_jobs':-1

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
lgb_params = {

    'n_estimators':800,

    'seed':seed

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
train_idx = activity[activity['data'] == 'train'].acc_id

test1_idx = activity[activity['data'] == 'test1'].acc_id

test2_idx = activity[activity['data'] == 'test2'].acc_id



train_set = data[data.acc_id.isin(train_idx)]

test1_set = data[data.acc_id.isin(test1_idx)]

test2_set = data[data.acc_id.isin(test2_idx)]



print('train set: ',train_set.shape)

print('test1 set: ',test1_set.shape)

print('test2 set: ',test2_set.shape)
train_set.fillna(0, inplace = True)

test1_set.fillna(0, inplace = True)

test2_set.fillna(0, inplace = True)
rf_params = {

    'n_estimators':1000,

    'max_depth':10,

    'n_jobs':-1

}
rf_amount = RandomForestRegressor(**rf_params)

rf_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

              train_set.amount_spent)

rf_amount_pred = rf_amount.predict(test1_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_amount_pred = pd.Series(rf_amount_pred).apply(lambda x: 0 if x < 0 else x)
rf_survival = RandomForestRegressor(**rf_params)

rf_survival.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                train_set.survival_time)

rf_survival_pred = rf_survival.predict(test1_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_survival_pred = pd.Series(rf_survival_pred).apply(lambda x: 64 if x > 64 else x)

rf_survival_pred = rf_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
rf_pred_df = pd.DataFrame({'acc_id':test1_set.acc_id.values,

                           'survival_time':rf_survival_pred,

                           'amount_spent':rf_amount_pred})

print('rf_pred_df shape: ',rf_pred_df.shape)
rf_pred_df.to_csv('test1_predict.csv', index = False) 
rf_amount = RandomForestRegressor(**rf_params)

rf_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

              train_set.amount_spent)

rf_amount_pred = rf_amount.predict(test2_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_amount_pred = pd.Series(rf_amount_pred).apply(lambda x: 0 if x < 0 else x)
rf_survival = RandomForestRegressor(**rf_params)

rf_survival.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                train_set.survival_time)

rf_survival_pred = rf_survival.predict(test2_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_survival_pred = pd.Series(rf_survival_pred).apply(lambda x: 64 if x > 64 else x)

rf_survival_pred = rf_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
rf_pred_df = pd.DataFrame({'acc_id':test2_set.acc_id.values,

                           'survival_time':rf_survival_pred,

                           'amount_spent':rf_amount_pred})

print('rf_pred_df shape: ',rf_pred_df.shape)
rf_pred_df.to_csv('test2_predict.csv', index = False) 