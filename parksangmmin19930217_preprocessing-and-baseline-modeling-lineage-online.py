import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

import warnings

warnings.filterwarnings('ignore')



from sklearn import metrics

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import RFE
train_label = pd.read_csv('../input/bigcontest2019/train_label.csv')

print("train_lable shape: {}".format(train_label.shape))
train_act = pd.read_csv('../input/bigcontest2019/train_activity.csv')

test1_act = pd.read_csv('../input/bigcontest2019/test1_activity.csv')

test2_act = pd.read_csv('../input/bigcontest2019/test2_activity.csv')



print("train_activity shape: {}".format(train_act.shape))

print("test1_activity shape: {}".format(test1_act.shape))

print("test2_activity shape: {}".format(test2_act.shape))



train_act['Train_or_Test'] = "Train"

test1_act['Train_or_Test'] = "Test1"

test2_act['Train_or_Test'] = "Test2"



all_act = pd.concat([train_act,test1_act,test2_act],axis= 0)
train_trade = pd.read_csv('../input/bigcontest2019/train_trade.csv')

test1_trade = pd.read_csv('../input/bigcontest2019/test1_trade.csv')

test2_trade = pd.read_csv('../input/bigcontest2019/test2_trade.csv')



print("train_trade shape: {}".format(train_trade.shape))

print("test1_trade shape: {}".format(test1_trade.shape))

print("test2_trade shape: {}".format(test2_trade.shape))



train_trade['Train_or_Test'] = "Train"

test1_trade['Train_or_Test'] = "Test1"

test2_trade['Train_or_Test'] = "Test2"



all_trade = pd.concat([train_trade, test1_trade, test2_trade], axis=0)
train_combat = pd.read_csv('../input/bigcontest2019/train_combat.csv')

test1_combat = pd.read_csv('../input/bigcontest2019/test1_combat.csv')

test2_combat = pd.read_csv('../input/bigcontest2019/test2_combat.csv')



print("train_combat.csv: {}".format(train_combat.shape))

print("test1_combat.csv: {}".format(test1_combat.shape))

print("test2_combat.csv: {}".format(test2_combat.shape))



train_combat['Train_or_Test'] = "Train"

test1_combat['Train_or_Test'] = 'Test1'

test2_combat['Train_or_Test'] = 'Test2'



all_combat = pd.concat([train_combat,test1_combat,test2_combat])
train_pledge = pd.read_csv('../input/bigcontest2019/train_pledge.csv')

test1_pledge = pd.read_csv('../input/bigcontest2019/test1_pledge.csv')

test2_pledge = pd.read_csv('../input/bigcontest2019/test2_pledge.csv')



print("train_pledge shape: {}".format(train_pledge.shape))

print("test1_pledge shape: {}".format(test1_pledge.shape))

print("test2_pledge shape: {}".format(test2_pledge.shape))



train_pledge['Train_or_Test'] = 'Train'

test1_pledge['Train_or_Test'] = 'Test1'

test2_pledge['Train_or_Test'] = 'Test2'



all_pledge = pd.concat([train_pledge,test1_pledge,test2_pledge])
train_pay = pd.read_csv('../input/bigcontest2019/train_payment.csv')

test1_pay = pd.read_csv('../input/bigcontest2019/test1_payment.csv')

test2_pay = pd.read_csv('../input/bigcontest2019/test2_payment.csv')



print("train payment shape : {}".format(train_pay.shape))

print("test1 payment shape : {}".format(test1_pay.shape))

print("test2 payment shape : {}".format(test2_pay.shape))



train_pay['Train_or_Test'] = 'Train'

test1_pay['Train_or_Test'] = 'Test1'

test2_pay['Train_or_Test'] = 'Test2'



all_pay = pd.concat([train_pay,test1_pay,test2_pay])
validation_acc = pd.read_csv('../input/bigcontest2019/train_valid_user_id.csv')
def make_week_variable(df):

    df['week'] = np.nan

    df['week'][df['day'] <= 7] = 1

    df['week'][(df['day']>7) & (df['day'] <= 14)] = 2

    df['week'][(df['day']>14) & (df['day'] <= 21)] = 3

    df['week'][df['day'] > 21] = 4

    print("Create Week Variable in DataFrame")
for i,name in enumerate([train_act,train_trade,train_combat,train_pledge,train_pay]):

    make_week_variable(name)

    print("No. {} Train DataFrame Success".format(i+1))

    print("\n")



for i,name in enumerate([test1_act,test1_trade,test1_combat,test1_pledge,test1_pay]):

    make_week_variable(name)

    print("No. {} Test1 DataFrame Success".format(i+1))

    print("\n")

    

for i,name in enumerate([test2_act,test2_trade,test2_combat,test2_pledge,test2_pay]):

    make_week_variable(name)

    print("No. {} Test2 DataFrame Success".format(i+1))

    print("\n")
train = train_act.groupby(['acc_id','server','char_id']).day.count().reset_index().groupby('acc_id').day.max().reset_index()

train.columns  =['acc_id','total_day']

train = pd.merge(train, train_label, on = 'acc_id', how = 'inner')



test1 = test1_act.groupby(['acc_id','server','char_id']).day.count().reset_index().groupby('acc_id').day.max().reset_index()

test1.columns = ['acc_id','total_day']

print(test1.shape)



test2 = test2_act.groupby(['acc_id','server','char_id']).day.count().reset_index().groupby('acc_id').day.max().reset_index()

test2.columns = ['acc_id','total_day']

print(test2.shape)
total_char_id = train_act[['acc_id','server','char_id']].drop_duplicates().groupby('acc_id').char_id.count().reset_index()

total_char_id.columns = ['acc_id','total_char']

train = pd.merge(train, total_char_id, on ='acc_id', how='left')



total_char_id = test1_act[['acc_id','server','char_id']].drop_duplicates().groupby('acc_id').char_id.count().reset_index()

total_char_id.columns = ['acc_id', 'total_char']

test1 = pd.merge(test1, total_char_id, on = 'acc_id', how = 'left')



total_char_id = test2_act[['acc_id','server','char_id']].drop_duplicates().groupby('acc_id').char_id.count().reset_index()

total_char_id.columns = ['acc_id','total_char']

test2 = pd.merge(test2, total_char_id, on = 'acc_id', how = 'left')
playtime_by_day = train_act.groupby(['day','acc_id']).playtime.sum().reset_index().groupby('acc_id').agg({'playtime':['std','mean']})

playtime_by_day.columns = ['std_playtime','avg_playtime']

train = pd.merge(train, playtime_by_day, on='acc_id', how='left')

train = train.fillna(0)



playtime_by_day = test1_act.groupby(['day','acc_id']).playtime.sum().reset_index().groupby('acc_id').agg({'playtime':['std','mean']})

playtime_by_day.columns = ['std_playtime','avg_playtime']

test1 = pd.merge(test1, playtime_by_day, on='acc_id', how='left')

test1 = test1.fillna(0)



playtime_by_day = test2_act.groupby(['day','acc_id']).playtime.sum().reset_index().groupby('acc_id').agg({'playtime':['std','mean']})

playtime_by_day.columns = ['std_playtime','avg_playtime']

test2 = pd.merge(test2, playtime_by_day, on='acc_id', how='left')

test2 = test2.fillna(0)
def make_variable_about_week(df,col,target_df):

    change = df[['acc_id','server','char_id',col,'week']].drop_duplicates().groupby(['acc_id','week'])[col].sum().unstack()

    change = change.fillna(0)

    change.columns = ['{}_week{}'.format(col,i+1) for i in range(4)]

    target_df = pd.merge(target_df, change, on = 'acc_id', how = 'left')

    

    return target_df.head()
party_exp_change = train_act[['acc_id','server','char_id','party_exp','week']].drop_duplicates().groupby(['acc_id','week']).party_exp.sum().unstack()

party_exp_change = party_exp_change.fillna(0)

party_exp_change.columns = ['party_exp_week{}'.format(i+1) for i in range(4)]

train = pd.merge(train, party_exp_change, on = 'acc_id',how = 'left')



party_exp_change = test1_act[['acc_id','server','char_id','party_exp','week']].drop_duplicates().groupby(['acc_id','week']).party_exp.sum().unstack()

party_exp_change = party_exp_change.fillna(0)

party_exp_change.columns = ['party_exp_week{}'.format(i+1) for i in range(4)]

test1 = pd.merge(test1, party_exp_change, on = 'acc_id', how = 'left')



party_exp_change = test2_act[['acc_id','server','char_id','party_exp','week']].drop_duplicates().groupby(['acc_id','week']).party_exp.sum().unstack()

party_exp_change = party_exp_change.fillna(0)

party_exp_change.columns = ['party_exp_week{}'.format(i+1) for i in range(4)]

test2 = pd.merge(test2, party_exp_change, on = 'acc_id', how = 'left')
train['change_party_exp'] = (train['party_exp_week4']+train['party_exp_week3'])/(train['party_exp_week2']+train['party_exp_week1'])

train.replace([np.inf, -np.inf], np.nan,inplace = True)

train = train.fillna(0)

train[['party_exp_week4','party_exp_week3','party_exp_week2','party_exp_week1','change_party_exp']].head()



test1['change_party_exp'] = (test1['party_exp_week4']+test1['party_exp_week3'])/(test1['party_exp_week2']+test1['party_exp_week1'])

test1.replace([np.inf, -np.inf],np.nan,inplace = True)

test1 = test1.fillna(0)



test2['change_party_exp'] = (test2['party_exp_week4']+test2['party_exp_week3'])/(test2['party_exp_week2']+test2['party_exp_week1'])

test2.replace([np.inf, -np.inf],np.nan,inplace = True)

test2 = test2.fillna(0)
death_change = train_act[['acc_id','server','char_id','death','week']].drop_duplicates().groupby(['acc_id','week']).death.sum().unstack()

death_change = death_change.fillna(0)

death_change.columns = ['death_cnt_week{}'.format(i+1) for i in range(4)]

train = pd.merge(train,death_change, on = 'acc_id', how = 'left')



death_change = test1_act[['acc_id','server','char_id','death','week']].drop_duplicates().groupby(['acc_id','week']).death.sum().unstack()

death_change = death_change.fillna(0)

death_change.columns = ['death_cnt_week{}'.format(i+1) for i in range(4)]

test1 = pd.merge(test1, death_change, on = 'acc_id', how = 'left')



death_change = test2_act[['acc_id','server','char_id','death','week']].drop_duplicates().groupby(['acc_id','week']).death.sum().unstack()

death_change = death_change.fillna(0)

death_change.columns = ['death_cnt_week{}'.format(i+1) for i in range(4)]

test2 = pd.merge(test2, death_change, on = 'acc_id', how ='left')
train['change_death_cnt'] = (train['death_cnt_week4']+train['death_cnt_week3'])/(train['death_cnt_week2']+train['death_cnt_week1'])

train.replace([np.inf, -np.inf], np.nan,inplace = True)

train = train.fillna(0)

train[['death_cnt_week4','death_cnt_week3','death_cnt_week2','death_cnt_week1','change_death_cnt']].head()



test1['change_death_cnt'] = (test1['death_cnt_week4']+test1['death_cnt_week3'])/(test1['death_cnt_week2']+test1['death_cnt_week1'])

test1.replace([np.inf, -np.inf], np.nan,inplace = True)

test1 = test1.fillna(0)



test2['change_death_cnt'] = (test2['death_cnt_week4']+test2['death_cnt_week3'])/(test2['death_cnt_week2']+test2['death_cnt_week1'])

test2.replace([np.inf, -np.inf], np.nan,inplace = True)

test2 = test2.fillna(0)
adena_change = train_act[['acc_id','server','char_id','game_money_change','week']].drop_duplicates().groupby(['acc_id','week']).game_money_change.sum().unstack()

adena_change = adena_change.fillna(0)

adena_change.columns = ['adena_change_week{}'.format(i+1) for i in range(4)]

train = pd.merge(train, adena_change, on = 'acc_id', how = 'left')



adena_change = test1_act[['acc_id','server','char_id','game_money_change','week']].drop_duplicates().groupby(['acc_id','week']).game_money_change.sum().unstack()

adena_change = adena_change.fillna(0)

adena_change.columns = ['adena_change_week{}'.format(i+1) for i in range(4)]

test1 = pd.merge(test1,adena_change, on = 'acc_id', how = 'left')



adena_change = test2_act[['acc_id','server','char_id','game_money_change','week']].drop_duplicates().groupby(['acc_id','week']).game_money_change.sum().unstack()

adena_change = adena_change.fillna(0)

adena_change.columns = ['adena_change_week{}'.format(i+1) for i in range(4)]

test2 = pd.merge(test2, adena_change, on = 'acc_id', how = 'left')
train['change_adena'] = (train['adena_change_week4']+train['adena_change_week3'])/(train['adena_change_week2']+train['adena_change_week1'])

train.replace([np.inf, -np.inf], np.nan,inplace = True)

train = train.fillna(0)

train[['adena_change_week4','adena_change_week3','adena_change_week2','adena_change_week1','change_adena']].head()



test1['change_adena'] = (test1['adena_change_week4']+test1['adena_change_week3'])/(test1['adena_change_week2']+test1['adena_change_week1'])

test1.replace([np.inf, -np.inf], np.nan, inplace = True)

test1 = test1.fillna(0)



test2['change_adena'] = (test2['adena_change_week4']+test2['adena_change_week3'])/(test2['adena_change_week2']+test2['adena_change_week1'])

test2.replace([np.inf, -np.inf], np.nan, inplace = True)

test2 = test2.fillna(0)
revive_change = train_act[['acc_id','server','char_id','revive','week']].drop_duplicates().groupby(['acc_id','week']).revive.sum().unstack()

revive_change = revive_change.fillna(0)

revive_change.columns = ['revive_change_week{}'.format(i+1) for i in range(4)]

train = pd.merge(train, revive_change, on = 'acc_id', how = 'left')



revive_change = test1_act[['acc_id','server','char_id','revive','week']].drop_duplicates().groupby(['acc_id','week']).revive.sum().unstack()

revive_change = revive_change.fillna(0)

revive_change.columns = ['revive_change_week{}'.format(i+1) for i in range(4)]

test1 = pd.merge(test1, revive_change, on = 'acc_id', how = 'left')



revive_change = test2_act[['acc_id','server','char_id','revive','week']].drop_duplicates().groupby(['acc_id','week']).revive.sum().unstack()

revive_change = revive_change.fillna(0)

revive_change.columns = ['revive_change_week{}'.format(i+1) for i in range(4)]

test2 = pd.merge(test2, revive_change, on ='acc_id', how = 'left')
train['change_revive'] = (train['revive_change_week4']+train['revive_change_week3'])/(train['revive_change_week2']+train['revive_change_week1'])

train.replace([np.inf, -np.inf], np.nan,inplace = True)

train = train.fillna(0)

train[['revive_change_week4','revive_change_week3','revive_change_week2','revive_change_week1','change_revive']].head()



test1['change_revive'] = (test1['revive_change_week4']+test1['revive_change_week3'])/(test1['revive_change_week2']+test1['revive_change_week1'])

test1.replace([np.inf, -np.inf], np.nan,inplace = True)

test1 = test1.fillna(0)



test2['change_revive'] = (test2['revive_change_week4']+test2['revive_change_week3'])/(test2['revive_change_week2']+test2['revive_change_week1'])

test2.replace([np.inf, -np.inf], np.nan,inplace = True)

test2 = test2.fillna(0)
private_change = train_act[['acc_id','server','char_id','private_shop','week']].drop_duplicates().groupby(['acc_id','week']).private_shop.sum().unstack()

private_change = private_change.fillna(0)

private_change.columns = ['private_shop_change_week{}'.format(i+1) for i in range(4)]

train = pd.merge(train,private_change, on = 'acc_id', how ='left')



private_change = test1_act[['acc_id','server','char_id','private_shop','week']].drop_duplicates().groupby(['acc_id','week']).private_shop.sum().unstack()

private_change = private_change.fillna(0)

private_change.columns = ['private_shop_change_week{}'.format(i+1) for i in range(4)]

test1 = pd.merge(test1,private_change, on = 'acc_id', how ='left')



private_change = test2_act[['acc_id','server','char_id','private_shop','week']].drop_duplicates().groupby(['acc_id','week']).private_shop.sum().unstack()

private_change = private_change.fillna(0)

private_change.columns = ['private_shop_change_week{}'.format(i+1) for i in range(4)]

test2 = pd.merge(test2,private_change, on = 'acc_id', how ='left')
test1.head()
train['change_private_shop'] = (train['private_shop_change_week4']+train['private_shop_change_week3'])/(train['private_shop_change_week2']+train['private_shop_change_week1'])

train.replace([np.inf, -np.inf], np.nan,inplace = True)

train = train.fillna(0)

train[['private_shop_change_week4','private_shop_change_week3','private_shop_change_week2','private_shop_change_week1','change_private_shop']].head()



test1['change_private_shop'] = (test1['private_shop_change_week4']+test1['private_shop_change_week3'])/(test1['private_shop_change_week2']+test1['private_shop_change_week1'])

test1.replace([np.inf, -np.inf], np.nan,inplace = True)

test1 = test1.fillna(0)



test2['change_private_shop'] = (test2['private_shop_change_week4']+test2['private_shop_change_week3'])/(test2['private_shop_change_week2']+test2['private_shop_change_week1'])

test2.replace([np.inf, -np.inf], np.nan,inplace = True)

test2 = test2.fillna(0)
del party_exp_change, death_change, adena_change, revive_change, private_change
all_trade['hour'] = all_trade['time'].str.split(':',expand = True)[0]



trade_buy = all_trade.copy()

trade_sell = all_trade.copy()



trade_buy['trade_type'] = 'Buy'

trade_sell['trade_type'] = 'Sell'



trade_buy.drop(['source_acc_id','source_char_id'], axis = 1, inplace= True)

trade_sell.drop(['target_acc_id','target_char_id'], axis = 1, inplace = True)



trade_buy = trade_buy.rename(columns = {'target_acc_id':'acc_id','target_char_id':'char_id'})

trade_sell = trade_sell.rename(columns = {'source_acc_id':'acc_id','source_char_id':'char_id'})
item_type_buy = pd.crosstab(trade_buy.loc[trade_buy['Train_or_Test']=='Train']['acc_id'], trade_buy.loc[trade_buy['Train_or_Test']=='Train']['item_type'])

item_type_buy.rename(columns = {'accessory':'get_accessory','adena':'get_adena',

                               'armor':'get_armor','enchant_scroll':'get_enchant_scroll',

                               'etc':'get_etc','spell':'get_spell','weapon':'get_weapon'},inplace= True)

train = pd.merge(train, item_type_buy, on = 'acc_id',how = 'left')





item_type_sell = pd.crosstab(trade_sell.loc[trade_buy['Train_or_Test']=='Train']['acc_id'], trade_sell.loc[trade_buy['Train_or_Test']=='Train']['item_type'])

item_type_sell.rename(columns = {'accessory':'put_accessory','adena':'put_adena',

                               'armor':'put_armor','enchant_scroll':'put_enchant_scroll',

                               'etc':'put_etc','spell':'put_spell','put':'put_weapon'},inplace=True)

train = pd.merge(train, item_type_sell, on = 'acc_id',how = 'left')



train = train.fillna(0)

train.head()



######################################################################



item_type_buy = pd.crosstab(trade_buy.loc[trade_buy['Train_or_Test']=='Test1']['acc_id'], trade_buy.loc[trade_buy['Train_or_Test'] == 'Test1']['item_type'])

item_type_buy.rename(columns = {'accessory':'get_accessory','adena':'get_adena',

                               'armor':'get_armor','enchant_scroll':'get_enchant_scroll',

                               'etc':'get_ect','spell':'get_spell','weapon':'get_weapon'},inplace= True)

test1 = pd.merge(test1, item_type_buy, on = 'acc_id', how = 'left')



item_type_sell = pd.crosstab(trade_sell.loc[trade_buy['Train_or_Test']=='Test1']['acc_id'], trade_sell.loc[trade_buy['Train_or_Test']=='Test1']['item_type'])

item_type_sell.rename(columns = {'accessory':'put_accessory','adena':'put_adena',

                               'armor':'put_armor','enchant_scroll':'put_enchant_scroll',

                               'etc':'put_etc','spell':'put_spell','put':'put_weapon'},inplace=True)

test1 = pd.merge(test1, item_type_sell, on = 'acc_id',how = 'left')



test1 = test1.fillna(0)

#

item_type_buy = pd.crosstab(trade_buy.loc[trade_buy['Train_or_Test']=='Test2']['acc_id'], trade_buy.loc[trade_buy['Train_or_Test'] == 'Test2']['item_type'])

item_type_buy.rename(columns = {'accessory':'get_accessory','adena':'get_adena',

                               'armor':'get_armor','enchant_scroll':'get_enchant_scroll',

                               'etc':'get_ect','spell':'get_spell','weapon':'get_weapon'},inplace= True)

test2 = pd.merge(test2, item_type_buy, on = 'acc_id', how = 'left')



item_type_sell = pd.crosstab(trade_sell.loc[trade_buy['Train_or_Test']=='Test2']['acc_id'], trade_sell.loc[trade_buy['Train_or_Test']=='Test2']['item_type'])

item_type_sell.rename(columns = {'accessory':'put_accessory','adena':'put_adena',

                               'armor':'put_armor','enchant_scroll':'put_enchant_scroll',

                               'etc':'put_etc','spell':'put_spell','put':'put_weapon'},inplace=True)

test2 = pd.merge(test2, item_type_sell, on = 'acc_id',how = 'left')



test2 = test2.fillna(0)



char_level = train_combat.groupby(['acc_id']).agg({'level': 'mean'})

char_level.columns = ['mean_level']

train = pd.merge(train, char_level, on = 'acc_id', how = 'left')



char_level = test1_combat.groupby(['acc_id']).agg({'level':'mean'})

char_level.columns = ['mean_level']

test1 = pd.merge(test1, char_level, on ='acc_id', how = 'left')



char_level = test2_combat.groupby(['acc_id']).agg({'level':'mean'})

char_level.columns = ['mean_level']

test2 = pd.merge(test2, char_level, on ='acc_id', how = 'left')
class_by_user = train_combat[['acc_id','server','char_id','class']].drop_duplicates().groupby(['acc_id','class']).char_id.count().unstack()

class_by_user = class_by_user.fillna(0)

class_by_user.columns = ['class{}'.format(i) for i in range(8)]

train = pd.merge(train, class_by_user, on = 'acc_id', how = 'left')



class_by_user = test1_combat[['acc_id','server','char_id','class']].drop_duplicates().groupby(['acc_id','class']).char_id.count().unstack()

class_by_user = class_by_user.fillna(0)

class_by_user.columns = ['class{}'.format(i) for i in range(8)]

test1 = pd.merge(test1, class_by_user, on='acc_id', how='left')



class_by_user = test2_combat[['acc_id','server','char_id','class']].drop_duplicates().groupby(['acc_id','class']).char_id.count().unstack()

class_by_user = class_by_user.fillna(0)

class_by_user.columns = ['class{}'.format(i) for i in range(8)]

test2 = pd.merge(test2, class_by_user, on='acc_id', how='left')
random_df = train_combat.groupby('acc_id').agg({'random_attacker_cnt':'sum','random_defender_cnt':'sum'})

random_df.columns = ['total_random_attacker_cnt','total_random_defender_cnt']

train = pd.merge(train, random_df, on='acc_id', how='left')



random_df = test1_combat.groupby('acc_id').agg({'random_attacker_cnt':'sum','random_defender_cnt':'sum'})

random_df.columns = ['total_random_attacker_cnt','total_random_defender_cnt']

test1 = pd.merge(test1, random_df, on='acc_id', how='left')



random_df = test2_combat.groupby('acc_id').agg({'random_attacker_cnt':'sum','random_defender_cnt':'sum'})

random_df.columns = ['total_random_attacker_cnt','total_random_defender_cnt']

test2 = pd.merge(test2, random_df, on='acc_id', how='left')
temp_df = train_combat[['acc_id','server','char_id','temp_cnt','week']].drop_duplicates().groupby(['acc_id','week']).temp_cnt.sum().unstack()

temp_df.columns = ['temp_cnt_week{}'.format(i+1) for i in range(4)]

temp_df = temp_df.fillna(0)

train = pd.merge(train,temp_df, on = 'acc_id', how = 'left')



temp_df = test1_combat[['acc_id','server','char_id','temp_cnt','week']].drop_duplicates().groupby(['acc_id','week']).temp_cnt.sum().unstack()

temp_df.columns = ['temp_cnt_week{}'.format(i+1) for i in range(4)]

temp_df = temp_df.fillna(0)

test1 = pd.merge(test1, temp_df, on = 'acc_id', how = 'left')



temp_df = test2_combat[['acc_id','server','char_id','temp_cnt','week']].drop_duplicates().groupby(['acc_id','week']).temp_cnt.sum().unstack()

temp_df.columns = ['temp_cnt_week{}'.format(i+1) for i in range(4)]

temp_df = temp_df.fillna(0)

test2 = pd.merge(test2, temp_df, on = 'acc_id', how = 'left')
pledge_df = train_combat[['acc_id','server','char_id','pledge_cnt','week']].drop_duplicates().groupby(['acc_id','week']).pledge_cnt.sum().unstack()

pledge_df.columns = ['pledge_cnt_week{}'.format(i+1) for i in range(4)]

pledge_df = pledge_df.fillna(0)

train = pd.merge(train,pledge_df, on = 'acc_id', how = 'left')



pledge_df = test1_combat[['acc_id','server','char_id','pledge_cnt','week']].drop_duplicates().groupby(['acc_id','week']).pledge_cnt.sum().unstack()

pledge_df.columns = ['pledge_cnt_week{}'.format(i+1) for i in range(4)]

pledge_df = pledge_df.fillna(0)

test1 = pd.merge(test1, pledge_df, on = 'acc_id', how = 'left')



pledge_df = test2_combat[['acc_id','server','char_id','pledge_cnt','week']].drop_duplicates().groupby(['acc_id','week']).pledge_cnt.sum().unstack()

pledge_df.columns = ['pledge_cnt_week{}'.format(i+1) for i in range(4)]

pledge_df = pledge_df.fillna(0)

test2 = pd.merge(test2, pledge_df, on = 'acc_id', how = 'left')
etc_df = train_combat[['acc_id','server','char_id','etc_cnt','week']].drop_duplicates().groupby(['acc_id','week']).etc_cnt.sum().unstack()

etc_df.columns = ['etc_cnt_week{}'.format(i+1) for i in range(4)]

etc_df = etc_df.fillna(0)

train = pd.merge(train,etc_df, on = 'acc_id', how = 'left')



etc_df = test1_combat[['acc_id','server','char_id','etc_cnt','week']].drop_duplicates().groupby(['acc_id','week']).etc_cnt.sum().unstack()

etc_df.columns = ['etc_cnt_week{}'.format(i+1) for i in range(4)]

etc_df = etc_df.fillna(0)

test1 = pd.merge(test1, etc_df, on = 'acc_id', how = 'left')



etc_df = test2_combat[['acc_id','server','char_id','etc_cnt','week']].drop_duplicates().groupby(['acc_id','week']).etc_cnt.sum().unstack()

etc_df.columns = ['etc_cnt_week{}'.format(i+1) for i in range(4)]

etc_df = etc_df.fillna(0)

test2 = pd.merge(test2, etc_df, on = 'acc_id', how = 'left')
combat_pledge_df = pd.merge(train_combat,train_pledge, on = ['day','acc_id','char_id','server'], how = 'left')

pledge_level_df = combat_pledge_df.groupby(['pledge_id','server']).agg({'level':'mean'})

pledge_level_df = pledge_level_df.fillna(0)

pledge_level_df.columns = ['pledge_mean_level']

train_pledge = pd.merge(train_pledge,pledge_level_df, on = 'pledge_id', how = 'left')



combat_pledge_df = pd.merge(test1_combat,test1_pledge, on = ['day','acc_id','char_id','server'], how = 'left')

pledge_level_df = combat_pledge_df.groupby(['pledge_id','server']).agg({'level':'mean'})

pledge_level_df = pledge_level_df.fillna(0)

pledge_level_df.columns = ['pledge_mean_level']

test1_pledge = pd.merge(test1_pledge,pledge_level_df, on = 'pledge_id', how = 'left')



combat_pledge_df = pd.merge(test2_combat,test2_pledge, on = ['day','acc_id','char_id','server'], how = 'left')

pledge_level_df = combat_pledge_df.groupby(['pledge_id','server']).agg({'level':'mean'})

pledge_level_df = pledge_level_df.fillna(0)

pledge_level_df.columns = ['pledge_mean_level']

test2_pledge = pd.merge(test2_pledge,pledge_level_df, on = 'pledge_id', how = 'left')
pledge_level_df = train_pledge.groupby('acc_id').agg({'pledge_mean_level':'mean'})

train = pd.merge(train,pledge_level_df, on = 'acc_id', how = 'left')

train = train.fillna(0)



pledge_level_df = test1_pledge.groupby('acc_id').agg({'pledge_mean_level':'mean'})

test1 = pd.merge(test1,pledge_level_df, on = 'acc_id', how = 'left')

test1 = test1.fillna(0)



pledge_level_df = test2_pledge.groupby('acc_id').agg({'pledge_mean_level':'mean'})

test2 = pd.merge(test2,pledge_level_df, on = 'acc_id', how = 'left')

test2 = test2.fillna(0)
train['pledge_belonging'] = train.mean_level / train.pledge_mean_level

train.replace([np.inf, -np.inf], np.nan,inplace = True)

train = train.fillna(0)



test1['pledge_belonging'] = test1.mean_level / test1.pledge_mean_level

test1.replace([np.inf,-np.inf], np.nan, inplace= True)

test1 = test1.fillna(0)



test2['pledge_belonging'] = test2.mean_level / test2.pledge_mean_level

test2.replace([np.inf, -np.inf], np.nan, inplace = True)

test2 = test2.fillna(0)
amount_by_acc_id = train_pay.groupby('acc_id').agg({'amount_spent':['max','median','sum']})

amount_by_acc_id.columns = ['max_amount','median_amount','total_amount']

train = pd.merge(train, amount_by_acc_id, on='acc_id', how='left')

train = train.fillna(0)



amount_by_acc_id = test1_pay.groupby('acc_id').agg({'amount_spent':['max','median','sum']})

amount_by_acc_id.columns = ['max_amount','median_amount','total_amount']

test1 = pd.merge(test1, amount_by_acc_id, on='acc_id', how='left')

test1 = test1.fillna(0)



amount_by_acc_id = test2_pay.groupby('acc_id').agg({'amount_spent':['max','median','sum']})

amount_by_acc_id.columns = ['max_amount','median_amount','total_amount']

test2 = pd.merge(test2, amount_by_acc_id, on='acc_id', how='left')

test2 = test2.fillna(0)
amount_by_acc_id = train_pay[['acc_id','week','amount_spent']].drop_duplicates().groupby(['acc_id','week']).amount_spent.sum().unstack()

amount_by_acc_id.columns = ['amount_spent_week{}'.format(i+1) for i in range(4)]

train = pd.merge(train, amount_by_acc_id, on = 'acc_id',how = 'left')

train = train.fillna(0)



amount_by_acc_id = test1_pay[['acc_id','week','amount_spent']].drop_duplicates().groupby(['acc_id','week']).amount_spent.sum().unstack()

amount_by_acc_id.columns = ['amount_spent_week{}'.format(i+1) for i in range(4)]

test1 = pd.merge(test1, amount_by_acc_id, on = 'acc_id', how = 'left')

test1 = test1.fillna(0)



amount_by_acc_id = test2_pay[['acc_id','week','amount_spent']].drop_duplicates().groupby(['acc_id','week']).amount_spent.sum().unstack()

amount_by_acc_id.columns = ['amount_spent_week{}'.format(i+1) for i in range(4)]

test2 = pd.merge(test2, amount_by_acc_id,on = 'acc_id',how = 'left')

test2 = test2.fillna(0)
train.to_csv('Train.csv',index = False)

test1.to_csv('Test1.csv',index = False)

test2.to_csv('Test2.csv',index = False)
train_index = validation_acc[validation_acc.set == 'Train'].acc_id

valid_index = validation_acc[validation_acc.set == 'Validation'].acc_id



train_set = train[train['acc_id'].isin(train_index)]

valid_set = train[train['acc_id'].isin(valid_index)]



print('Train set:',train_set.shape)

print('Valid set:',valid_set.shape)
def survival64(y_pred, dataset):

    y_true = dataset.get_label()

    y_pred = np.array([64 if x > 64 else x for x in y_pred])

    y_pred = np.array([0 if x < 0 else x for x in y_pred])

    y_pred = np.round(y_pred)

    error = np.sqrt(mean_squared_error(y_true, y_pred))

    return 'error', error, False
lgb_params = {'n_estimators': 1000,

             'seed': 123}



lgb_train_amount = lgb.Dataset(train_set.drop(['acc_id','amount_spent','survival_time'],axis = 1),

                              train_set.amount_spent)

lgb_train_survival = lgb.Dataset(train_set.drop(['acc_id','amount_spent','survival_time'],axis = 1),

                                train_set.survival_time)
lgb_amount = lgb.train(lgb_params,

                      lgb_train_amount,

                      feval = survival64,

                      valid_sets = [lgb_train_amount],

                      verbose_eval=100)



lgb_amount_pred = lgb_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lgb_amount_pred = pd.Series(lgb_amount_pred).apply(lambda x: 0 if x < 0 else x)
lgb_survival = lgb.train(lgb_params,

                        lgb_train_survival,

                        feval =survival64,

                        valid_sets = [lgb_train_survival],

                        verbose_eval = 100)



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
lgb_valid_score = score_function(lgb_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

print('Light GBM score: ',lgb_valid_score)