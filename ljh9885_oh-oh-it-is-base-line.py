import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')

from sklearn.preprocessing import scale, minmax_scale

import os

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor as rf

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

print(train_valid.shape)
test1_act = pd.read_csv('../input/bigcontest2019/test1_activity.csv')

test2_act = pd.read_csv('../input/bigcontest2019/test2_activity.csv')

test1_com = pd.read_csv('../input/bigcontest2019/test1_combat.csv')

test2_com = pd.read_csv('../input/bigcontest2019/test2_combat.csv')

test1_ple = pd.read_csv('../input/bigcontest2019/test1_pledge.csv')

test2_ple = pd.read_csv('../input/bigcontest2019/test2_pledge.csv')

test1_tra = pd.read_csv('../input/bigcontest2019/test1_trade.csv')

test2_tra = pd.read_csv('../input/bigcontest2019/test2_trade.csv')

test1_pay = pd.read_csv('../input/bigcontest2019/test1_payment.csv')

test2_pay = pd.read_csv('../input/bigcontest2019/test2_payment.csv')

print('test1_activity.shape :',test1_act.shape)

print('test1_combat.shape :',test1_com.shape)

print('test1_pledge.shape :',test1_ple.shape)

print('test1_trade.shape :',test1_tra.shape)

print('test1_payment.shape :',test1_pay.shape)

print("----------------------------------------")

print('test2_activity.shape :',test2_act.shape)

print('test2_combat.shape :',test2_com.shape)

print('test2_pledge.shape :',test2_ple.shape)

print('test2_trade.shape :',test2_tra.shape)

print('test2_payment.shape :',test2_pay.shape)
# train_act = train_act[train_act['day']>=8]

# train_com = train_com[train_com['day']>=8]

# train_ple = train_ple[train_ple['day']>=8]

# train_tra = train_tra[train_tra['day']>=8]

# train_pay = train_pay[train_pay['day']>=8]

# 실험체
ple = train_ple.drop(['acc_id','char_id'],axis=1)

ple.head()
ple2 = ple.drop_duplicates()

ple2.shape
ple2['pledge_id'] = ple2['pledge_id'].astype(str)

ple2['real_pledge_id'] = ple2['server'] + '_' + ple2['pledge_id']

ple2.drop(['server','pledge_id'],axis=1,inplace=True)

ple2.head()
ple2.drop('non_combat_play_time',axis=1,inplace=True)
ple2['real_pledge_id'].nunique()
ple2.columns
ple2 = ple2.groupby('real_pledge_id')[['play_char_cnt','combat_char_cnt',

                                       'pledge_combat_cnt','random_attacker_cnt',

                                       'random_defender_cnt','same_pledge_cnt',

                                       'temp_cnt','etc_cnt',

                                       'combat_play_time']].sum().reset_index()

print(ple2.shape)

ple2.head()
sns.lmplot(x='combat_char_cnt',y='combat_play_time',data=ple2,fit_reg=False)
dp = ple2.drop(['real_pledge_id'],axis=1).values
def elbow(x):

    sse = []

    for i in range(1,16):

        km = KMeans(n_clusters=i, random_state=19)

        km.fit(x)

        sse.append(km.inertia_)

    

    plt.plot(range(1,16), sse, marker='o')

    plt.xlabel('k')

    plt.ylabel('SSE')

    plt.show()



elbow(dp)
km = KMeans(n_clusters=3)

km.fit(dp)

km.labels_[:5]
ple2['cluster'] = km.labels_

ple2.head()
sns.lmplot(x='play_char_cnt',y='combat_char_cnt',data=ple2,fit_reg=False,hue='cluster')
ple2 = ple2[['real_pledge_id','cluster']]

ple2.head()
ple2['cluster'].value_counts()
train_ple['pledge_id2'] = train_ple['pledge_id'].astype(str)

train_ple['real_pledge_id'] = train_ple['server'] + '_' + train_ple['pledge_id2']

train_ple.drop('pledge_id2',axis=1,inplace=True)

train_ple.head()
test1_ple['pledge_id2'] = test1_ple['pledge_id'].astype(str)

test1_ple['real_pledge_id'] = test1_ple['server'] + '_' + test1_ple['pledge_id2']

test1_ple.drop('pledge_id2',axis=1,inplace=True)

test1_ple.head()
test2_ple['pledge_id2'] = test2_ple['pledge_id'].astype(str)

test2_ple['real_pledge_id'] = test2_ple['server'] + '_' + test2_ple['pledge_id2']

test2_ple.drop('pledge_id2',axis=1,inplace=True)

test2_ple.head()
# train_ple = pd.merge(train_ple, ple2, on='real_pledge_id', how='left')
train_act.head()
act0 = train_act.groupby('acc_id')[['day','char_id','server']].nunique().reset_index()

act1 = test1_act.groupby('acc_id')[['day','char_id','server']].nunique().reset_index()

act2 = test2_act.groupby('acc_id')[['day','char_id','server']].nunique().reset_index()

act0.head()
ps0 = train_act.groupby('acc_id')['playtime'].sum().reset_index(name='playtime_sum')

pm0 = train_act.groupby('acc_id')['playtime'].mean().reset_index(name='playtime_mean')

pst0 = train_act.groupby('acc_id')['playtime'].std(ddof=0).reset_index(name='playtime_std')

act0_1 = pd.merge(ps0, pm0, on='acc_id', how='left')

act0_1 = pd.merge(act0_1, pst0, on='acc_id', how='left')

ps1 = test1_act.groupby('acc_id')['playtime'].sum().reset_index(name='playtime_sum')

pm1 = test1_act.groupby('acc_id')['playtime'].mean().reset_index(name='playtime_mean')

pst1 = test1_act.groupby('acc_id')['playtime'].std(ddof=0).reset_index(name='playtime_std')

act1_1 = pd.merge(ps1, pm1, on='acc_id', how='left')

act1_1 = pd.merge(act1_1, pst1, on='acc_id', how='left')

ps2 = test2_act.groupby('acc_id')['playtime'].sum().reset_index(name='playtime_sum')

pm2 = test2_act.groupby('acc_id')['playtime'].mean().reset_index(name='playtime_mean')

pst2 = test2_act.groupby('acc_id')['playtime'].std(ddof=0).reset_index(name='playtime_std')

act2_1 = pd.merge(ps2, pm2, on='acc_id', how='left')

act2_1 = pd.merge(act2_1, pst2, on='acc_id', how='left')

act0_1.head()
train_act['game_money'] = abs(train_act['game_money_change'])

test1_act['game_money'] = abs(test1_act['game_money_change'])

test2_act['game_money'] = abs(test2_act['game_money_change'])

act0_2 = train_act.groupby('acc_id')[['npc_kill','solo_exp','quest_exp','party_exp','fishing','death','revive','exp_recovery','enchant_count','game_money','rich_monster']].sum().reset_index()

act1_2 = test1_act.groupby('acc_id')[['npc_kill','solo_exp','quest_exp','party_exp','fishing','death','revive','exp_recovery','enchant_count','game_money','rich_monster']].sum().reset_index()

act2_2 = test2_act.groupby('acc_id')[['npc_kill','solo_exp','quest_exp','party_exp','fishing','death','revive','exp_recovery','enchant_count','game_money','rich_monster']].sum().reset_index()

act0_2.head()
ss0 = train_act.groupby('acc_id')['private_shop'].sum().reset_index(name='private_shop_sum')

sm0 = train_act.groupby('acc_id')['private_shop'].mean().reset_index(name='private_shop_mean')

act0_3 = pd.merge(ss0, sm0, on='acc_id', how='left')

ss1 = test1_act.groupby('acc_id')['private_shop'].sum().reset_index(name='private_shop_sum')

sm1 = test1_act.groupby('acc_id')['private_shop'].mean().reset_index(name='private_shop_mean')

act1_3 = pd.merge(ss1, sm1, on='acc_id', how='left')

ss2 = test2_act.groupby('acc_id')['private_shop'].sum().reset_index(name='private_shop_sum')

sm2 = test2_act.groupby('acc_id')['private_shop'].mean().reset_index(name='private_shop_mean')

act2_3 = pd.merge(ss2, sm2, on='acc_id', how='left')

act0_3.head()
activity0 = pd.merge(act0, act0_1, on='acc_id', how='left')

activity0 = pd.merge(activity0, act0_2, on='acc_id', how='left')

activity0 = pd.merge(activity0, act0_3, on='acc_id', how='left')



activity1 = pd.merge(act1, act1_1, on='acc_id', how='left')

activity1 = pd.merge(activity1, act1_2, on='acc_id', how='left')

activity1 = pd.merge(activity1, act1_3, on='acc_id', how='left')



activity2 = pd.merge(act2, act2_1, on='acc_id', how='left')

activity2 = pd.merge(activity2, act2_2, on='acc_id', how='left')

activity2 = pd.merge(activity2, act2_3, on='acc_id', how='left')

activity0.head()
activity1.head()
activity2.head()
train_com.head()
cd0 = pd.get_dummies(train_com['class'], prefix='class')

cd1 = pd.get_dummies(test1_com['class'], prefix='class')

cd2 = pd.get_dummies(test2_com['class'], prefix='class')

cd0.head()
train_com = train_com.join(cd0)

test1_com = test1_com.join(cd1)

test2_com = test2_com.join(cd2)

train_com.head()
com0 = train_com.groupby('acc_id')[['class_0','class_1','class_2','class_3','class_4','class_5','class_6','class_7']].sum().reset_index()

com1 = test1_com.groupby('acc_id')[['class_0','class_1','class_2','class_3','class_4','class_5','class_6','class_7']].sum().reset_index()

com2 = test2_com.groupby('acc_id')[['class_0','class_1','class_2','class_3','class_4','class_5','class_6','class_7']].sum().reset_index()

com0.head()
com0_1 = train_com.groupby('acc_id')['level'].mean().reset_index()

com1_1 = test1_com.groupby('acc_id')['level'].mean().reset_index()

com2_1 = test2_com.groupby('acc_id')['level'].mean().reset_index()

com0_1.head()
cps0 = train_com.groupby('acc_id')['pledge_cnt'].sum().reset_index(name='pledge_cnt_sum')

cpm0 = train_com.groupby('acc_id')['pledge_cnt'].mean().reset_index(name='pledge_cnt_mean')

com0_2 = pd.merge(cps0, cpm0, on='acc_id', how='left')

cps1 = test1_com.groupby('acc_id')['pledge_cnt'].sum().reset_index(name='pledge_cnt_sum')

cpm1 = test1_com.groupby('acc_id')['pledge_cnt'].mean().reset_index(name='pledge_cnt_mean')

com1_2 = pd.merge(cps1, cpm1, on='acc_id', how='left')

cps2 = test2_com.groupby('acc_id')['pledge_cnt'].sum().reset_index(name='pledge_cnt_sum')

cpm2 = test2_com.groupby('acc_id')['pledge_cnt'].mean().reset_index(name='pledge_cnt_mean')

com2_2 = pd.merge(cps2, cpm2, on='acc_id', how='left')

com0_2.head()
crs0 = train_com.groupby('acc_id')['random_attacker_cnt'].sum().reset_index(name='random_attacker_cnt_sum')

crm0 = train_com.groupby('acc_id')['random_attacker_cnt'].mean().reset_index(name='random_attacker_cnt_mean')

com0_3 = pd.merge(crs0, crm0, on='acc_id', how='left')

crs1 = test1_com.groupby('acc_id')['random_attacker_cnt'].sum().reset_index(name='random_attacker_cnt_sum')

crm1 = test1_com.groupby('acc_id')['random_attacker_cnt'].mean().reset_index(name='random_attacker_cnt_mean')

com1_3 = pd.merge(crs1, crm1, on='acc_id', how='left')

crs2 = test2_com.groupby('acc_id')['random_attacker_cnt'].sum().reset_index(name='random_attacker_cnt_sum')

crm2 = test2_com.groupby('acc_id')['random_attacker_cnt'].mean().reset_index(name='random_attacker_cnt_mean')

com2_3 = pd.merge(crs2, crm2, on='acc_id', how='left')

com0_3.head()
crds0 = train_com.groupby('acc_id')['random_defender_cnt'].sum().reset_index(name='random_defender_cnt_sum')

crdm0 = train_com.groupby('acc_id')['random_defender_cnt'].mean().reset_index(name='random_defender_cnt_mean')

com0_4 = pd.merge(crds0, crdm0, on='acc_id', how='left')

crds1 = test1_com.groupby('acc_id')['random_defender_cnt'].sum().reset_index(name='random_defender_cnt_sum')

crdm1 = test1_com.groupby('acc_id')['random_defender_cnt'].mean().reset_index(name='random_defender_cnt_mean')

com1_4 = pd.merge(crds1, crdm1, on='acc_id', how='left')

crds2 = test2_com.groupby('acc_id')['random_defender_cnt'].sum().reset_index(name='random_defender_cnt_sum')

crdm2 = test2_com.groupby('acc_id')['random_defender_cnt'].mean().reset_index(name='random_defender_cnt_mean')

com2_4 = pd.merge(crds2, crdm2, on='acc_id', how='left')

com0_4.head()
cts0 = train_com.groupby('acc_id')['temp_cnt'].sum().reset_index(name='temp_cnt_sum')

ctm0 = train_com.groupby('acc_id')['temp_cnt'].mean().reset_index(name='temp_cnt_mean')

com0_5 = pd.merge(cts0, ctm0, on='acc_id', how='left')

cts1 = test1_com.groupby('acc_id')['temp_cnt'].sum().reset_index(name='temp_cnt_sum')

ctm1 = test1_com.groupby('acc_id')['temp_cnt'].mean().reset_index(name='temp_cnt_mean')

com1_5 = pd.merge(cts1, ctm1, on='acc_id', how='left')

cts2 = test2_com.groupby('acc_id')['temp_cnt'].sum().reset_index(name='temp_cnt_sum')

ctm2 = test2_com.groupby('acc_id')['temp_cnt'].mean().reset_index(name='temp_cnt_mean')

com2_5 = pd.merge(cts2, ctm2, on='acc_id', how='left')

com0_5.head()
css0 = train_com.groupby('acc_id')['same_pledge_cnt'].sum().reset_index(name='same_pledge_cnt_sum')

csm0 = train_com.groupby('acc_id')['same_pledge_cnt'].mean().reset_index(name='same_pledge_cnt_mean')

com0_6 = pd.merge(css0, csm0, on='acc_id', how='left')

css1 = test1_com.groupby('acc_id')['same_pledge_cnt'].sum().reset_index(name='same_pledge_cnt_sum')

csm1 = test1_com.groupby('acc_id')['same_pledge_cnt'].mean().reset_index(name='same_pledge_cnt_mean')

com1_6 = pd.merge(css1, csm1, on='acc_id', how='left')

css2 = test2_com.groupby('acc_id')['same_pledge_cnt'].sum().reset_index(name='same_pledge_cnt_sum')

csm2 = test2_com.groupby('acc_id')['same_pledge_cnt'].mean().reset_index(name='same_pledge_cnt_mean')

com2_6 = pd.merge(css2, csm2, on='acc_id', how='left')

com0_6.head()
ces0 = train_com.groupby('acc_id')['etc_cnt'].sum().reset_index(name='etc_cnt_sum')

cem0 = train_com.groupby('acc_id')['etc_cnt'].mean().reset_index(name='etc_cnt_mean')

com0_7 = pd.merge(ces0, cem0, on='acc_id', how='left')

ces1 = test1_com.groupby('acc_id')['etc_cnt'].sum().reset_index(name='etc_cnt_sum')

cem1 = test1_com.groupby('acc_id')['etc_cnt'].mean().reset_index(name='etc_cnt_mean')

com1_7 = pd.merge(ces1, cem1, on='acc_id', how='left')

ces2 = test2_com.groupby('acc_id')['etc_cnt'].sum().reset_index(name='etc_cnt_sum')

cem2 = test2_com.groupby('acc_id')['etc_cnt'].mean().reset_index(name='etc_cnt_mean')

com2_7 = pd.merge(ces2, cem2, on='acc_id', how='left')

com0_7.head()
cns0 = train_com.groupby('acc_id')['num_opponent'].sum().reset_index(name='num_opponent_sum')

cnm0 = train_com.groupby('acc_id')['num_opponent'].mean().reset_index(name='num_opponent_mean')

com0_8 = pd.merge(cns0, cnm0, on='acc_id', how='left')

cns1 = test1_com.groupby('acc_id')['num_opponent'].sum().reset_index(name='num_opponent_sum')

cnm1 = test1_com.groupby('acc_id')['num_opponent'].mean().reset_index(name='num_opponent_mean')

com1_8 = pd.merge(cns1, cnm1, on='acc_id', how='left')

cns2 = test2_com.groupby('acc_id')['num_opponent'].sum().reset_index(name='num_opponent_sum')

cnm2 = test2_com.groupby('acc_id')['num_opponent'].mean().reset_index(name='num_opponent_mean')

com2_8 = pd.merge(cns2, cnm2, on='acc_id', how='left')

com0_8.head()
combat0 = pd.merge(com0, com0_1, on='acc_id', how='left')

combat0 = pd.merge(combat0, com0_2, on='acc_id', how='left')

combat0 = pd.merge(combat0, com0_3, on='acc_id', how='left')

combat0 = pd.merge(combat0, com0_4, on='acc_id', how='left')

combat0 = pd.merge(combat0, com0_5, on='acc_id', how='left')

combat0 = pd.merge(combat0, com0_6, on='acc_id', how='left')

combat0 = pd.merge(combat0, com0_7, on='acc_id', how='left')

combat0 = pd.merge(combat0, com0_8, on='acc_id', how='left')



combat1 = pd.merge(com1, com1_1, on='acc_id', how='left')

combat1 = pd.merge(combat1, com1_2, on='acc_id', how='left')

combat1 = pd.merge(combat1, com1_3, on='acc_id', how='left')

combat1 = pd.merge(combat1, com1_4, on='acc_id', how='left')

combat1 = pd.merge(combat1, com1_5, on='acc_id', how='left')

combat1 = pd.merge(combat1, com1_6, on='acc_id', how='left')

combat1 = pd.merge(combat1, com1_7, on='acc_id', how='left')

combat1 = pd.merge(combat1, com1_8, on='acc_id', how='left')



combat2 = pd.merge(com2, com2_1, on='acc_id', how='left')

combat2 = pd.merge(combat2, com2_2, on='acc_id', how='left')

combat2 = pd.merge(combat2, com2_3, on='acc_id', how='left')

combat2 = pd.merge(combat2, com2_4, on='acc_id', how='left')

combat2 = pd.merge(combat2, com2_5, on='acc_id', how='left')

combat2 = pd.merge(combat2, com2_6, on='acc_id', how='left')

combat2 = pd.merge(combat2, com2_7, on='acc_id', how='left')

combat2 = pd.merge(combat2, com2_8, on='acc_id', how='left')



combat0.head()
combat1.head()
combat2.head()
train_ple.head()
# pds = pd.get_dummies(train_ple['cluster'],prefix='pledge_clust')

# pds.head()
# train_ple = train_ple.join(pds)

# train_ple.drop('cluster',axis=1,inplace=True)

# train_ple.head()
# pdc = train_ple.groupby('acc_id')[['pledge_clust_0','pledge_clust_1','pledge_clust_2']].sum().reset_index()

# pdc.head()
# pdc['pledge_clust_0'] = pdc['pledge_clust_0'].map(lambda x: 1 if x>0 else 0)

# pdc['pledge_clust_1'] = pdc['pledge_clust_1'].map(lambda x: 1 if x>0 else 0)

# pdc['pledge_clust_2'] = pdc['pledge_clust_2'].map(lambda x: 1 if x>0 else 0)

# pdc.head()
pledge0 = train_ple.groupby(['acc_id'])['real_pledge_id'].nunique().reset_index()

pledge0.head()
pledge1 = test1_ple.groupby(['acc_id'])['real_pledge_id'].nunique().reset_index()

pledge1.head()
pledge2 = test2_ple.groupby(['acc_id'])['real_pledge_id'].nunique().reset_index()

pledge2.head()
train_tra.head()
train_sell = train_tra.drop(['target_acc_id','target_char_id'],axis=1)  # 판매 데이터

train_buy = train_tra.drop(['source_acc_id','source_char_id'],axis=1)   # 구매 데이터



test1_sell = test1_tra.drop(['target_acc_id','target_char_id'],axis=1)  # 판매 데이터

test1_buy = test1_tra.drop(['source_acc_id','source_char_id'],axis=1)   # 구매 데이터



test2_sell = test2_tra.drop(['target_acc_id','target_char_id'],axis=1)  # 판매 데이터

test2_buy = test2_tra.drop(['source_acc_id','source_char_id'],axis=1)   # 구매 데이터

train_sell.head()
train_buy.head()
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

train_sell.head()
train_buy.head()
uniq_acc_id0 = train_lab['acc_id'].values

print(len(uniq_acc_id0))

uniq_acc_id0[:10]
uniq_acc_id1 = test1_act['acc_id'].unique()

print(len(uniq_acc_id1))

uniq_acc_id1[:10]
uniq_acc_id2 = test2_act['acc_id'].unique()

print(len(uniq_acc_id2))

uniq_acc_id2[:10]
train_sell = train_sell[train_sell['acc_id'].isin(uniq_acc_id0)]

train_buy = train_buy[train_buy['acc_id'].isin(uniq_acc_id0)]



test1_sell = test1_sell[test1_sell['acc_id'].isin(uniq_acc_id1)]

test1_buy = test1_buy[test1_buy['acc_id'].isin(uniq_acc_id1)]



test2_sell = test2_sell[test2_sell['acc_id'].isin(uniq_acc_id2)]

test2_buy = test2_buy[test2_buy['acc_id'].isin(uniq_acc_id2)]

print(train_sell.shape)

train_sell.head()
print(train_buy.shape)

train_buy.head()
sid0 = pd.get_dummies(train_sell['sell_item_type'],prefix='sell')

bid0 = pd.get_dummies(train_buy['buy_item_type'],prefix='buy')



sid1 = pd.get_dummies(test1_sell['sell_item_type'],prefix='sell')

bid1 = pd.get_dummies(test1_buy['buy_item_type'],prefix='buy')



sid2 = pd.get_dummies(test2_sell['sell_item_type'],prefix='sell')

bid2 = pd.get_dummies(test2_buy['buy_item_type'],prefix='buy')



sid0.head()
train_sell = train_sell.join(sid0)

train_buy = train_buy.join(bid0)

test1_sell = test1_sell.join(sid1)

test1_buy = test1_buy.join(bid1)

test2_sell = test2_sell.join(sid2)

test2_buy = test2_buy.join(bid2)

train_sell.head()
si0 = train_sell.groupby('acc_id')[['sell_accessory','sell_adena','sell_armor','sell_enchant_scroll','sell_etc','sell_spell','sell_weapon']].sum().reset_index()

bi0 = train_buy.groupby('acc_id')[['buy_accessory','buy_adena','buy_armor','buy_enchant_scroll','buy_etc','buy_spell','buy_weapon']].sum().reset_index()



si1 = test1_sell.groupby('acc_id')[['sell_accessory','sell_adena','sell_armor','sell_enchant_scroll','sell_etc','sell_spell','sell_weapon']].sum().reset_index()

bi1 = test1_buy.groupby('acc_id')[['buy_accessory','buy_adena','buy_armor','buy_enchant_scroll','buy_etc','buy_spell','buy_weapon']].sum().reset_index()



si2 = test2_sell.groupby('acc_id')[['sell_accessory','sell_adena','sell_armor','sell_enchant_scroll','sell_etc','sell_spell','sell_weapon']].sum().reset_index()

bi2 = test2_buy.groupby('acc_id')[['buy_accessory','buy_adena','buy_armor','buy_enchant_scroll','buy_etc','buy_spell','buy_weapon']].sum().reset_index()

si0.head()
bi0.head()
std0 = pd.get_dummies(train_sell['sell_type'],prefix='sell')

btd0 = pd.get_dummies(train_buy['buy_type'],prefix='buy')



std1 = pd.get_dummies(test1_sell['sell_type'],prefix='sell')

btd1 = pd.get_dummies(test1_buy['buy_type'],prefix='buy')



std2 = pd.get_dummies(test2_sell['sell_type'],prefix='sell')

btd2 = pd.get_dummies(test2_buy['buy_type'],prefix='buy')



std0.head()
train_sell = train_sell.join(std0)

train_buy = train_buy.join(btd0)

test1_sell = test1_sell.join(std1)

test1_buy = test1_buy.join(btd1)

test2_sell = test2_sell.join(std2)

test2_buy = test2_buy.join(btd2)

train_sell.head()
st0 = train_sell.groupby('acc_id')[['sell_0','sell_1']].sum().reset_index()

bt0 = train_buy.groupby('acc_id')[['buy_0','buy_1']].sum().reset_index()



st1 = test1_sell.groupby('acc_id')[['sell_0','sell_1']].sum().reset_index()

bt1 = test1_buy.groupby('acc_id')[['buy_0','buy_1']].sum().reset_index()



st2 = test2_sell.groupby('acc_id')[['sell_0','sell_1']].sum().reset_index()

bt2 = test2_buy.groupby('acc_id')[['buy_0','buy_1']].sum().reset_index()

st0.head()
sas0 = train_sell.groupby('acc_id')['sell_item_amount'].sum().reset_index()

bas0 = train_buy.groupby('acc_id')['buy_item_amount'].sum().reset_index()



sas1 = test1_sell.groupby('acc_id')['sell_item_amount'].sum().reset_index()

bas1 = test1_buy.groupby('acc_id')['buy_item_amount'].sum().reset_index()



sas2 = test2_sell.groupby('acc_id')['sell_item_amount'].sum().reset_index()

bas2 = test2_buy.groupby('acc_id')['buy_item_amount'].sum().reset_index()

sas0.head()
trade0 = pd.merge(si0, bi0, on='acc_id', how='left')

trade0 = pd.merge(trade0, st0, on='acc_id', how='left')

trade0 = pd.merge(trade0, bt0, on='acc_id', how='left')

trade0 = pd.merge(trade0, sas0, on='acc_id', how='left')

trade0 = pd.merge(trade0, bas0, on='acc_id', how='left')



trade1 = pd.merge(si1, bi1, on='acc_id', how='left')

trade1 = pd.merge(trade1, st1, on='acc_id', how='left')

trade1 = pd.merge(trade1, bt1, on='acc_id', how='left')

trade1 = pd.merge(trade1, sas1, on='acc_id', how='left')

trade1 = pd.merge(trade1, bas1, on='acc_id', how='left')



trade2 = pd.merge(si2, bi2, on='acc_id', how='left')

trade2 = pd.merge(trade2, st2, on='acc_id', how='left')

trade2 = pd.merge(trade2, bt2, on='acc_id', how='left')

trade2 = pd.merge(trade2, sas2, on='acc_id', how='left')

trade2 = pd.merge(trade2, bas2, on='acc_id', how='left')

trade0.head()
train_pay.head()
tp0 = train_pay.groupby('acc_id')['day'].nunique().reset_index(name='spent_day_count')

tp1 = test1_pay.groupby('acc_id')['day'].nunique().reset_index(name='spent_day_count')

tp2 = test2_pay.groupby('acc_id')['day'].nunique().reset_index(name='spent_day_count')

tp0.head()
sds0 = train_pay.groupby('acc_id')['amount_spent'].sum().reset_index(name='sum_spent')

sds1 = test1_pay.groupby('acc_id')['amount_spent'].sum().reset_index(name='sum_spent')

sds2 = test2_pay.groupby('acc_id')['amount_spent'].sum().reset_index(name='sum_spent')

sds0.head()
ds0 = pd.merge(act0[['acc_id','day']], sds0, on='acc_id')

ds1 = pd.merge(act1[['acc_id','day']], sds1, on='acc_id')

ds2 = pd.merge(act2[['acc_id','day']], sds2, on='acc_id')

ds0.head()
ds0['daily_spent'] = ds0['sum_spent'] / ds0['day']

ds1['daily_spent'] = ds1['sum_spent'] / ds1['day']

ds2['daily_spent'] = ds2['sum_spent'] / ds2['day']

ds0.drop('day',axis=1,inplace=True)

ds1.drop('day',axis=1,inplace=True)

ds2.drop('day',axis=1,inplace=True)

ds0.head()
payment0 = pd.merge(tp0, ds0, on='acc_id', how='left')



payment1 = pd.merge(tp1, ds1, on='acc_id', how='left')



payment2 = pd.merge(tp2, ds2, on='acc_id', how='left')

payment0.head()
train = pd.merge(activity0, combat0, on='acc_id', how='left')

train = pd.merge(train, pledge0, on='acc_id', how='left')

train = pd.merge(train, trade0, on='acc_id', how='left')

train = pd.merge(train, payment0, on='acc_id', how='left')

train = train.fillna(0)

train = pd.merge(train, train_lab, on='acc_id', how='left')

print(train.shape)

train.head()
test1 = pd.merge(activity1, combat1, on='acc_id', how='left')

test1 = pd.merge(test1, pledge1, on='acc_id', how='left')

test1 = pd.merge(test1, trade1, on='acc_id', how='left')

test1 = pd.merge(test1, payment1, on='acc_id', how='left')

test1 = test1.fillna(0)

print(test1.shape)

test1.head()
test2 = pd.merge(activity2, combat2, on='acc_id', how='left')

test2 = pd.merge(test2, pledge2, on='acc_id', how='left')

test2 = pd.merge(test2, trade2, on='acc_id', how='left')

test2 = pd.merge(test2, payment2, on='acc_id', how='left')

test2 = test2.fillna(0)

print(test2.shape)

test2.head()
tt = train.copy()

tt = pd.merge(tt, train_valid, on='acc_id', how='left')

tt.head()
# tt['is_survival'] = 0

# tt['is_survival'][tt['survival_time']==64] = 1

# tt['is_spent'] = 0

# tt['is_spent'][tt['amount_spent']>0] = 1

# tt.head()
tt_train = tt[tt['set']=='Train']

tt_valid = tt[tt['set']=='Validation']

print(tt_train.shape, tt_valid.shape)
tt_train1 = tt_train[tt_train['survival_time']<64]

tt_train2 = tt_train[tt_train['amount_spent']>0]

print(tt_train1.shape)

print(tt_train2.shape)
x_tt_train1 = tt_train1.drop(['acc_id','survival_time','amount_spent','set'],axis=1)

x_tt_train2 = tt_train2.drop(['acc_id','survival_time','amount_spent','set'],axis=1)

x_tt_valid = tt_valid.drop(['acc_id','survival_time','amount_spent','set'],axis=1)

y1_tt_train = tt_train1['survival_time']

y2_tt_train = tt_train2['amount_spent']
xgb_params={'eta':0.01,

            'max_depth':6,

            'objective':'reg:squarederror',

            'eval_metric':'mae',

            'min_child_samples':1,

            'tree_method':'gpu_hist',

            'predictor':'gpu_predictor'}
tt_dtrain = xgb.DMatrix(x_tt_train1, y1_tt_train)

tt_dtest = xgb.DMatrix(x_tt_valid)



cv_output = xgb.cv(xgb_params,

                   tt_dtrain,

                   num_boost_round = 5000,

                   nfold = 5,

                   early_stopping_rounds = 50,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds1 = cv_output.index.size



print('Best rounds :',best_rounds1)
model1 = xgb.train(xgb_params,

                   tt_dtrain,

                   num_boost_round = best_rounds1)

y1_tt_pred = model1.predict(tt_dtest)

y1_tt_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model1,ax=ax)

plt.show()
tt_dtrain = xgb.DMatrix(x_tt_train2, y2_tt_train)

tt_dtest = xgb.DMatrix(x_tt_valid)



cv_output = xgb.cv(xgb_params,

                   tt_dtrain,

                   num_boost_round = 5000,

                   nfold = 5,

                   early_stopping_rounds = 50,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds2 = cv_output.index.size



print('Best rounds :',best_rounds2)
model2 = xgb.train(xgb_params,

                   tt_dtrain,

                   num_boost_round = best_rounds1)

y2_tt_pred = model2.predict(tt_dtest)

y2_tt_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model2,ax=ax)

plt.show()
tt_tt = pd.DataFrame({'acc_id':tt_valid['acc_id'],

                      'survival_time':y1_tt_pred,

                      'amount_spent':y2_tt_pred})

tt_tt.head()
tt_tt['survival_time'] = tt_tt['survival_time'].apply(lambda x: 64 if x>64 else x)

tt_tt['survival_time'] = tt_tt['survival_time'].apply(lambda x: 1 if x<1 else x).round()

tt_tt['amount_spent'] = tt_tt['amount_spent'].apply(lambda x: 0 if x<0 else x)

tt_tt.head()
# x_cla_train = tt_train.drop(['acc_id','survival_time','amount_spent','set','is_survival','is_spent'],axis=1)

# x_cla_valid = tt_valid.drop(['acc_id','survival_time','amount_spent','set','is_survival','is_spent'],axis=1)

# y1_cla_train = tt_train['is_survival']

# y2_cla_train = tt_train['is_spent']
# xgb_params={'eta':0.05,

#             'max_depth':6,

#             'objective':'binary:logistic',

#             'eval_metric':'auc',

#             'min_child_samples':2,

#             'tree_method':'gpu_hist',

#             'predictor':'gpu_predictor'}
# cla_dtrain = xgb.DMatrix(x_cla_train,y1_cla_train)

# cla_dtest = xgb.DMatrix(x_cla_valid)



# cv_output = xgb.cv(xgb_params,

#                    cla_dtrain,

#                    num_boost_round = 5000,

#                    nfold = 5,

#                    early_stopping_rounds = 50,

#                    verbose_eval = 100,

#                    show_stdv = False)



# best_rounds = cv_output.index.size



# print('Best rounds :',best_rounds)
# model = xgb.train(xgb_params,

#                   cla_dtrain,

#                   num_boost_round = best_rounds)

# y1_cla_pred = model.predict(cla_dtest)

# y1_cla_pred[:5]
# tt_valid['surv_cla_pred'] = y1_cla_pred

# tt_valid['surv_cla_pred'] = tt_valid['surv_cla_pred'].map(lambda x: 0 if x<0.5 else 1)

# tt_valid.head()
# tt_train2 = tt_train[tt_train['survival_time']<64]

# tt_train2.head()
# tt_valid2 = tt_valid[tt_valid['surv_cla_pred']==0]

# tt_valid2.head()
# x_reg_train = tt_train2.drop(['acc_id','survival_time','amount_spent','set','is_survival','is_spent'],axis=1)

# x_reg_valid = tt_valid2.drop(['acc_id','survival_time','amount_spent','set','is_survival','is_spent','surv_cla_pred'],axis=1)

# y1_reg_train = tt_train2['survival_time']
# xgb_params={'eta':0.05,

#             'max_depth':6,

#             'objective':'reg:squarederror',

#             'eval_metric':'mae',

#             'min_child_samples':2,

#             'tree_method':'gpu_hist',

#             'predictor':'gpu_predictor'}
# reg_dtrain = xgb.DMatrix(x_reg_train, y1_reg_train)

# reg_dtest = xgb.DMatrix(x_reg_valid)



# cv_output = xgb.cv(xgb_params,

#                    reg_dtrain,

#                    num_boost_round = 5000,

#                    nfold = 5,

#                    early_stopping_rounds = 50,

#                    verbose_eval = 100,

#                    show_stdv = False)



# best_rounds = cv_output.index.size



# print('Best rounds :',best_rounds)
# model = xgb.train(xgb_params,

#                   reg_dtrain,

#                   num_boost_round = best_rounds)

# y1_reg_pred = model.predict(reg_dtest)

# y1_reg_pred[:5]
# tt_valid2['survival_time_pred'] = y1_reg_pred

# tt_valid2['survival_time_pred'] = tt_valid2['survival_time_pred'].map(lambda x: 1 if x<0 else x)

# tt_valid2['survival_time_pred'] = tt_valid2['survival_time_pred'].map(lambda x: 63 if x>63 else x).round()

# tt_valid2.head()
# prediction = pd.merge(tt_valid[['acc_id']], tt_valid2[['acc_id','survival_time_pred']], on='acc_id', how='left')

# prediction['survival_time_pred'] = prediction['survival_time_pred'].fillna(64)

# prediction.head()
# xgb_params={'eta':0.05,

#             'max_depth':6,

#             'objective':'binary:logistic',

#             'eval_metric':'auc',

#             'min_child_samples':2,

#             'tree_method':'gpu_hist',

#             'predictor':'gpu_predictor'}
# cla_dtrain = xgb.DMatrix(x_cla_train, y2_cla_train)

# cla_dtest = xgb.DMatrix(x_cla_valid)



# cv_output = xgb.cv(xgb_params,

#                    cla_dtrain,

#                    num_boost_round = 5000,

#                    nfold = 5,

#                    early_stopping_rounds = 50,

#                    verbose_eval = 100,

#                    show_stdv = False)



# best_rounds = cv_output.index.size



# print('Best rounds :',best_rounds)
# model = xgb.train(xgb_params,

#                   cla_dtrain,

#                   num_boost_round = best_rounds)

# y2_cla_pred = model.predict(cla_dtest)

# y2_cla_pred[:5]
# tt_valid['amou_cla_pred'] = y2_cla_pred

# tt_valid['amou_cla_pred'] = tt_valid['amou_cla_pred'].map(lambda x: 0 if x<0.5 else 1)

# tt_valid.head()
# tt_train3 = tt_train[tt_train['amount_spent']>0]

# tt_train3.head()
# tt_valid3 = tt_valid[tt_valid['amou_cla_pred']==1]

# tt_valid3.head()
# x_reg_train = tt_train3.drop(['acc_id','survival_time','amount_spent','set','is_survival','is_spent'],axis=1)

# x_reg_valid = tt_valid3.drop(['acc_id','survival_time','amount_spent','set','is_survival','is_spent','surv_cla_pred','amou_cla_pred'],axis=1)

# y2_reg_train = tt_train3['amount_spent']
# xgb_params={'eta':0.05,

#             'max_depth':6,

#             'objective':'reg:squarederror',

#             'eval_metric':'mae',

#             'min_child_samples':2,

#             'tree_method':'gpu_hist',

#             'predictor':'gpu_predictor'}
# reg_dtrain = xgb.DMatrix(x_reg_train, y2_reg_train)

# reg_dtest = xgb.DMatrix(x_reg_valid)



# cv_output = xgb.cv(xgb_params,

#                    reg_dtrain,

#                    num_boost_round = 5000,

#                    nfold = 5,

#                    early_stopping_rounds = 50,

#                    verbose_eval = 100,

#                    show_stdv = False)



# best_rounds = cv_output.index.size



# print('Best rounds :',best_rounds)
# model = xgb.train(xgb_params,

#                   reg_dtrain,

#                   num_boost_round = best_rounds)

# y2_reg_pred = model.predict(reg_dtest)

# y2_reg_pred[:5]
# tt_valid3['amount_spent_pred'] = y2_reg_pred

# tt_valid3['amount_spent_pred'] = tt_valid3['amount_spent_pred'].map(lambda x: y2_reg_pred[y2_reg_pred>0].min() if x<0 else x)

# tt_valid3.head()
# prediction = pd.merge(prediction, tt_valid3[['acc_id','amount_spent_pred']], on='acc_id', how='left')

# prediction['amount_spent_pred'] = prediction['amount_spent_pred'].fillna(0)

# prediction.head()
# prediction = prediction.rename(columns = {'survival_time_pred':'survival_time',

#                                           'amount_spent_pred':'amount_spent'})

# prediction.head()
true = tt_valid[['acc_id','survival_time','amount_spent']]

true.head()
# tt_tt = pd.DataFrame({'acc_id':tt_valid['acc_id'],

#                       'survival_time':y1_tt_pred,

#                       'amount_spent':y2_tt_pred})

# tt_tt.head()
# def ss(x):

#     if x>64:

#         return 64

#     elif x<1:

#         return 1

#     else:

#         return x
# tt_tt['survival_time'] = tt_tt['survival_time'].map(ss)

# tt_tt['survival_time'] = np.round(tt_tt['survival_time'])

# tt_tt['amount_spent'] = tt_tt['amount_spent'].map(lambda x: 0 if x<0 else x)

# tt_tt.head()
# ttt = pd.merge(train_lab, train_valid, on='acc_id', how='left')

# true_ = ttt[ttt['set']=='Validation']

# true_.drop('set',axis=1,inplace=True)

# true_.head()
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

            

        score = sum(profit_result)                                         #기대이익 총합 계산

    return score
pred_test = tt_tt.copy()
pred_test['survival_time'] = 13

pred_test['amount_spent'] = 1
tt_test = tt_tt.copy()

from sklearn.preprocessing import minmax_scale

tt_test['amount_spent'] = minmax_scale(tt_test['amount_spent'], (0,95))

tt_test['survival_time'] = minmax_scale(tt_test['survival_time'], (1,30)).round()

tt_test.head()
mine_score = score_function(tt_tt, true)

oing_score = score_function(tt_test, true)

pred_test_score = score_function(pred_test, true)

true_score = score_function(true, true)



print('mine score :', mine_score)

print('뽕맛 score :', pred_test_score)

print('뽕맛 치사량 score :', oing_score)

print('true score :', true_score)
for i in [j/10 for j in range(1,11)]:

    print('{} :'.format(i), tt_tt['amount_spent'].quantile(i))
for i in [j/10 for j in range(1,11)]:

    print('{} :'.format(i), tt_test['amount_spent'].quantile(i))
for i in [j/10 for j in range(1,11)]:

    print('{} :'.format(i), train_lab['amount_spent'].quantile(i))
train1 = train[train['survival_time']!=64]

train2 = train[train['amount_spent']>0]

print(train1.shape)

print(train2.shape)

train1.head()
x_train1 = train1.drop(['acc_id','survival_time','amount_spent'],axis=1)

x_train2 = train2.drop(['acc_id','survival_time','amount_spent'],axis=1)

y1_train = train1['survival_time']

y2_train = train2['amount_spent']

x_test1 = test1.drop('acc_id',axis=1)

x_test2 = test2.drop('acc_id',axis=1)
xgb_params={'eta':0.01,

            'max_depth':6,

            'objective':'reg:squarederror',

            'eval_metric':'mae',

            'min_child_samples':1,

            'tree_method':'gpu_hist',

            'predictor':'gpu_predictor'}
dtrain = xgb.DMatrix(x_train1, y1_train)

dtest = xgb.DMatrix(x_test1)



cv_output = xgb.cv(xgb_params,

                   dtrain,

                   num_boost_round = 5000,

                   nfold = 5,

                   early_stopping_rounds = 50,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds1 = cv_output.index.size



print('Best rounds :',best_rounds1)
model1 = xgb.train(xgb_params,

                   dtrain,

                   num_boost_round = best_rounds1)

t1_y1_pred = model1.predict(dtest)

t1_y1_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model1,ax=ax)

plt.show()
dtrain = xgb.DMatrix(x_train2, y2_train)

dtest = xgb.DMatrix(x_test1)



cv_output = xgb.cv(xgb_params,

                   dtrain,

                   num_boost_round = 5000,

                   nfold = 5,

                   early_stopping_rounds = 50,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds2 = cv_output.index.size



print('Best rounds :',best_rounds2)
model2 = xgb.train(xgb_params,

                   dtrain,

                   num_boost_round = best_rounds2)

t1_y2_pred = model2.predict(dtest)

t1_y2_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model2,ax=ax)

plt.show()
test1_pred = pd.DataFrame({'acc_id':test1['acc_id'],

                           'survival_time':t1_y1_pred,

                           'amount_spent':t1_y2_pred})

test1_pred.head()
test1_pred['survival_time'] = test1_pred['survival_time'].apply(lambda x: 64 if x>64 else x)

test1_pred['survival_time'] = test1_pred['survival_time'].apply(lambda x: 1 if x<1 else x).round()

test1_pred['amount_spent'] = test1_pred['amount_spent'].map(lambda x: 0 if x<0 else x)

test1_pred['amount_spent'] = minmax_scale(test1_pred['amount_spent'], (0,95))

test1_pred['survival_time'] = minmax_scale(test1_pred['survival_time'], (1,30)).round()

test1_pred.head()
test1_pred.to_csv('test1_predict.csv', index=False)
dtrain = xgb.DMatrix(x_train1, y1_train)

dtest = xgb.DMatrix(x_test2)



cv_output = xgb.cv(xgb_params,

                   dtrain,

                   num_boost_round = 5000,

                   nfold = 5,

                   early_stopping_rounds = 50,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds1 = cv_output.index.size



print('Best rounds :',best_rounds1)
model1 = xgb.train(xgb_params,

                   dtrain,

                   num_boost_round = best_rounds1)

t2_y1_pred = model1.predict(dtest)

t2_y1_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model1,ax=ax)

plt.show()
dtrain = xgb.DMatrix(x_train2, y2_train)

dtest = xgb.DMatrix(x_test2)



cv_output = xgb.cv(xgb_params,

                   dtrain,

                   num_boost_round = 5000,

                   nfold = 5,

                   early_stopping_rounds = 50,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds2 = cv_output.index.size



print('Best rounds :',best_rounds2)
model2 = xgb.train(xgb_params,

                   dtrain,

                   num_boost_round = best_rounds2)

t2_y2_pred = model2.predict(dtest)

t2_y2_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model2,ax=ax)

plt.show()
test2_pred = pd.DataFrame({'acc_id':test2['acc_id'],

                           'survival_time':t2_y1_pred,

                           'amount_spent':t2_y2_pred})

test2_pred.head()
test2_pred['survival_time'] = test2_pred['survival_time'].apply(lambda x: 64 if x>64 else x)

test2_pred['survival_time'] = test2_pred['survival_time'].apply(lambda x: 1 if x<1 else x).round()

test2_pred['amount_spent'] = test2_pred['amount_spent'].map(lambda x: 0 if x<0 else x)

test2_pred['amount_spent'] = minmax_scale(test2_pred['amount_spent'], (0,95))

test2_pred['survival_time'] = minmax_scale(test2_pred['survival_time'], (1,30)).round()

test2_pred.head()
test2_pred.to_csv('test2_predict.csv', index=False)