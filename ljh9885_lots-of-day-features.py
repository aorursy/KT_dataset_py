import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib

matplotlib.rcParams['axes.unicode_minus']=False

matplotlib.rcParams['font.family']="Hancom Gothic"

plt.style.use('ggplot')

from sklearn.preprocessing import scale,minmax_scale

import os

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans
train_lab = pd.read_csv('../input/bigcontest2019/train_label.csv')

train_act = pd.read_csv('../input/bigcontest2019/train_activity.csv')

train_com = pd.read_csv('../input/bigcontest2019/train_combat.csv')

train_ple = pd.read_csv('../input/bigcontest2019/train_pledge.csv')

train_tra = pd.read_csv('../input/bigcontest2019/train_trade.csv')

train_pay = pd.read_csv('../input/bigcontest2019/train_payment.csv')

print('train_label.shape :',train_lab.shape)

print('train_activity.shape :',train_act.shape)

print('train_combat.shape :',train_com.shape)

print('train_pledge.shape :',train_ple.shape)

print('train_trade.shape :',train_tra.shape)

print('train_payment.shape :',train_pay.shape)
train_valid = pd.read_csv('../input/bigcontest2019/train_valid_user_id.csv')

print('train_vaild.shape :',train_valid.shape)
train_act['playtime'][(train_act['playtime']==0)&(train_act['fishing']!=0)] = train_act['fishing'][(train_act['playtime']==0)&(train_act['fishing']!=0)]

train_act[(train_act['playtime']==0)]
train_act.head()
train_com.head()
train_ple.head()
train_tra.head()
train_pay.head()
train = pd.DataFrame({'acc_id':train_act['acc_id'].unique()})

print(train.shape)

train.head()
train_act.columns
train_act['game_money_change'] = np.abs(train_act['game_money_change'])
day_playtime = train_act.groupby(['acc_id','day'])['playtime'].sum().unstack()

day_playtime = day_playtime.fillna(0)

day_playtime.columns = ['playtime_day{}'.format(i+1) for i in range(28)]

day_playtime = day_playtime.reset_index()

print(day_playtime.shape)

train = pd.merge(train, day_playtime, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_npc_kill = train_act.groupby(['acc_id','day'])['npc_kill'].sum().unstack()

day_npc_kill = day_npc_kill.fillna(0)

day_npc_kill.columns = ['npc_kill_day{}'.format(i+1) for i in range(28)]

day_npc_kill = day_npc_kill.reset_index()

print(day_npc_kill.shape)

train = pd.merge(train, day_npc_kill, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_solo_exp = train_act.groupby(['acc_id','day'])['solo_exp'].sum().unstack()

day_solo_exp = day_solo_exp.fillna(0)

day_solo_exp.columns = ['solo_exp_day{}'.format(i+1) for i in range(28)]

day_solo_exp = day_solo_exp.reset_index()

print(day_solo_exp.shape)

train = pd.merge(train, day_solo_exp, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_party_exp = train_act.groupby(['acc_id','day'])['party_exp'].sum().unstack()

day_party_exp = day_party_exp.fillna(0)

day_party_exp.columns = ['party_exp_day{}'.format(i+1) for i in range(28)]

day_party_exp = day_party_exp.reset_index()

print(day_party_exp.shape)

train = pd.merge(train, day_party_exp, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_quest_exp = train_act.groupby(['acc_id','day'])['quest_exp'].sum().unstack()

day_quest_exp = day_quest_exp.fillna(0)

day_quest_exp.columns = ['quest_exp_day{}'.format(i+1) for i in range(28)]

day_quest_exp = day_quest_exp.reset_index()

print(day_quest_exp.shape)

train = pd.merge(train, day_quest_exp, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_rich_monster = train_act.groupby(['acc_id','day'])['rich_monster'].sum().unstack()

day_rich_monster = day_rich_monster.fillna(0)

day_rich_monster.columns = ['rich_monster_day{}'.format(i+1) for i in range(28)]

day_rich_monster = day_rich_monster.reset_index()

print(day_rich_monster.shape)

train = pd.merge(train, day_rich_monster, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_death = train_act.groupby(['acc_id','day'])['death'].sum().unstack()

day_death = day_death.fillna(0)

day_death.columns = ['death_day{}'.format(i+1) for i in range(28)]

day_death = day_death.reset_index()

print(day_death.shape)

train = pd.merge(train, day_death, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_revive = train_act.groupby(['acc_id','day'])['revive'].sum().unstack()

day_revive = day_revive.fillna(0)

day_revive.columns = ['revive_day{}'.format(i+1) for i in range(28)]

day_revive = day_revive.reset_index()

print(day_revive.shape)

train = pd.merge(train, day_revive, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_exp_recovery = train_act.groupby(['acc_id','day'])['exp_recovery'].sum().unstack()

day_exp_recovery = day_exp_recovery.fillna(0)

day_exp_recovery.columns = ['exp_recovery_day{}'.format(i+1) for i in range(28)]

day_exp_recovery = day_exp_recovery.reset_index()

print(day_exp_recovery.shape)

train = pd.merge(train, day_exp_recovery, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_game_money = train_act.groupby(['acc_id','day'])['game_money_change'].sum().unstack()

day_game_money = day_game_money.fillna(0)

day_game_money.columns = ['game_money_change_day{}'.format(i+1) for i in range(28)]

day_game_money = day_game_money.reset_index()

print(day_game_money.shape)

train = pd.merge(train, day_game_money, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_fishing = train_act.groupby(['acc_id','day'])['fishing'].sum().unstack()

day_fishing = day_fishing.fillna(0)

day_fishing.columns = ['fishing_day{}'.format(i+1) for i in range(28)]

day_fishing = day_fishing.reset_index()

print(day_fishing.shape)

train = pd.merge(train, day_fishing, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_private_shop = train_act.groupby(['acc_id','day'])['private_shop'].sum().unstack()

day_private_shop = day_private_shop.fillna(0)

day_private_shop.columns = ['private_shop_day{}'.format(i+1) for i in range(28)]

day_private_shop = day_private_shop.reset_index()

print(day_private_shop.shape)

train = pd.merge(train, day_private_shop, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_enchant_count = train_act.groupby(['acc_id','day'])['enchant_count'].sum().unstack()

day_enchant_count = day_enchant_count.fillna(0)

day_enchant_count.columns = ['enchant_count_day{}'.format(i+1) for i in range(28)]

day_enchant_count = day_enchant_count.reset_index()

print(day_enchant_count.shape)

train = pd.merge(train, day_enchant_count, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
train_com.columns
day_level = train_com.groupby(['acc_id','day'])['level'].sum().unstack()

day_level = day_level.fillna(0)

day_level.columns = ['level_day{}'.format(i+1) for i in range(28)]

day_level = day_level.reset_index()

print(day_level.shape)

train = pd.merge(train, day_level, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_pledge_cnt = train_com.groupby(['acc_id','day'])['pledge_cnt'].sum().unstack()

day_pledge_cnt = day_pledge_cnt.fillna(0)

day_pledge_cnt.columns = ['pledge_cnt_day{}'.format(i+1) for i in range(28)]

day_pledge_cnt = day_pledge_cnt.reset_index()

print(day_pledge_cnt.shape)

train = pd.merge(train, day_pledge_cnt, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_random_attacker_cnt = train_com.groupby(['acc_id','day'])['random_attacker_cnt'].sum().unstack()

day_random_attacker_cnt = day_random_attacker_cnt.fillna(0)

day_random_attacker_cnt.columns = ['random_attacker_cnt_day{}'.format(i+1) for i in range(28)]

day_random_attacker_cnt = day_random_attacker_cnt.reset_index()

print(day_random_attacker_cnt.shape)

train = pd.merge(train, day_random_attacker_cnt, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_random_defender_cnt = train_com.groupby(['acc_id','day'])['random_defender_cnt'].sum().unstack()

day_random_defender_cnt = day_random_defender_cnt.fillna(0)

day_random_defender_cnt.columns = ['random_defender_cnt_day{}'.format(i+1) for i in range(28)]

day_random_defender_cnt = day_random_defender_cnt.reset_index()

print(day_random_defender_cnt.shape)

train = pd.merge(train, day_random_defender_cnt, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_temp_cnt = train_com.groupby(['acc_id','day'])['temp_cnt'].sum().unstack()

day_temp_cnt = day_temp_cnt.fillna(0)

day_temp_cnt.columns = ['temp_cnt_day{}'.format(i+1) for i in range(28)]

day_temp_cnt = day_temp_cnt.reset_index()

print(day_temp_cnt.shape)

train = pd.merge(train, day_temp_cnt, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_same_pledge_cnt = train_com.groupby(['acc_id','day'])['same_pledge_cnt'].sum().unstack()

day_same_pledge_cnt = day_same_pledge_cnt.fillna(0)

day_same_pledge_cnt.columns = ['same_pledge_cnt_day{}'.format(i+1) for i in range(28)]

day_same_pledge_cnt = day_same_pledge_cnt.reset_index()

print(day_same_pledge_cnt.shape)

train = pd.merge(train, day_same_pledge_cnt, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_etc_cnt = train_com.groupby(['acc_id','day'])['etc_cnt'].sum().unstack()

day_etc_cnt = day_etc_cnt.fillna(0)

day_etc_cnt.columns = ['etc_cnt_day{}'.format(i+1) for i in range(28)]

day_etc_cnt = day_etc_cnt.reset_index()

print(day_etc_cnt.shape)

train = pd.merge(train, day_etc_cnt, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_num_opponent = train_com.groupby(['acc_id','day'])['num_opponent'].sum().unstack()

day_num_opponent = day_num_opponent.fillna(0)

day_num_opponent.columns = ['num_opponent_day{}'.format(i+1) for i in range(28)]

day_num_opponent = day_num_opponent.reset_index()

print(day_num_opponent.shape)

train = pd.merge(train, day_num_opponent, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
train_sell = train_tra.drop(['target_acc_id','target_char_id'],axis=1)  # 판매 데이터

train_buy = train_tra.drop(['source_acc_id','source_char_id'],axis=1)   # 구매 데이터



test1_sell = test1_tra.drop(['target_acc_id','target_char_id'],axis=1)  # 판매 데이터

test1_buy = test1_tra.drop(['source_acc_id','source_char_id'],axis=1)   # 구매 데이터



test2_sell = test2_tra.drop(['target_acc_id','target_char_id'],axis=1)  # 판매 데이터

test2_buy = test2_tra.drop(['source_acc_id','source_char_id'],axis=1)   # 구매 데이터



train_sell = train_sell.rename(columns = {'source_acc_id':'acc_id',

                                          'source_char_id':'char_id',

                                          'item_type':'sell_item_type',

                                          'item_amount':'sell_item_amount',

                                          'item_price':'sell_item_price',

                                          'time':'sell_time',

                                          'type':'sell_type'})

train_buy = train_buy.rename(columns = {'target_acc_id':'acc_id',

                                         'target_char_id':'char_id',

                                         'item_type':'buy_item_type',

                                         'item_amount':'buy_item_amount',

                                         'item_price':'buy_item_price',

                                         'time':'buy_time',

                                         'type':'buy_type'})

test1_sell = test1_sell.rename(columns = {'source_acc_id':'acc_id',

                                          'source_char_id':'char_id',

                                          'item_type':'sell_item_type',

                                          'item_amount':'sell_item_amount',

                                          'item_price':'sell_item_price',

                                          'time':'sell_time',

                                          'type':'sell_type'})

test1_buy = test1_buy.rename(columns = {'target_acc_id':'acc_id',

                                        'target_char_id':'char_id',

                                        'item_type':'buy_item_type',

                                        'item_amount':'buy_item_amount',

                                        'item_price':'buy_item_price',

                                        'time':'buy_time',

                                        'type':'buy_type'})

test2_sell = test2_sell.rename(columns = {'source_acc_id':'acc_id',

                                          'source_char_id':'char_id',

                                          'item_type':'sell_item_type',

                                          'item_amount':'sell_item_amount',

                                          'item_price':'sell_item_price',

                                          'time':'sell_time',

                                          'type':'sell_type'})

test2_buy = test2_buy.rename(columns = {'target_acc_id':'acc_id',

                                        'target_char_id':'char_id',

                                        'item_type':'buy_item_type',

                                        'item_amount':'buy_item_amount',

                                        'item_price':'buy_item_price',

                                        'time':'buy_time',

                                        'type':'buy_type'})



uniq_acc_id0 = train_lab['acc_id'].values

uniq_acc_id1 = test1_act['acc_id'].unique()

uniq_acc_id2 = test2_act['acc_id'].unique()



train_sell = train_sell[train_sell['acc_id'].isin(uniq_acc_id0)]

train_buy = train_buy[train_buy['acc_id'].isin(uniq_acc_id0)]



test1_sell = test1_sell[test1_sell['acc_id'].isin(uniq_acc_id1)]

test1_buy = test1_buy[test1_buy['acc_id'].isin(uniq_acc_id1)]



test2_sell = test2_sell[test2_sell['acc_id'].isin(uniq_acc_id2)]

test2_buy = test2_buy[test2_buy['acc_id'].isin(uniq_acc_id2)]
train_pay.columns
day_amount_spent = train_pay.groupby(['acc_id','day'])['amount_spent'].sum().unstack()

day_amount_spent = day_amount_spent.fillna(0)

day_amount_spent.columns = ['amount_spent_day{}'.format(i+1) for i in range(28)]

day_amount_spent = day_amount_spent.reset_index()

print(day_amount_spent.shape)

train = pd.merge(train, day_amount_spent, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
day_day = train_pay.groupby(['acc_id','day'])['day'].count().unstack()

day_day = day_day.fillna(0)

day_day.columns = ['payment_cnt_day{}'.format(i+1) for i in range(28)]

day_day = day_day.reset_index()

print(day_day.shape)

train = pd.merge(train, day_day, on='acc_id', how='left')



print('train.shape :',train.shape)

train.head()
train = train.fillna(0)
train = pd.merge(train, train_lab, on='acc_id', how='left')

print('train.shape :',train.shape)
train.to_csv('day_train.csv', index=False)