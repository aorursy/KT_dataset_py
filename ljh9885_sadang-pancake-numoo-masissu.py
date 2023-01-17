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

import lightgbm as lgb

import xgboost as xgb
os.listdir('../input/')
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary=pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary=summary.reset_index()

    summary['Name']=summary['index']

    summary=summary[['Name','dtypes']]

    summary['Min']=df.min().values

    summary['Max']=df.max().values

    summary['Missing']=df.isnull().sum().values    

    summary['Uniques']=df.nunique().values

    return summary
train_lab=pd.read_csv('../input/bigcontest2019/train_label.csv')

print('train_lab.shape :',train_lab.shape)

train_lab.head()
resumetable(train_lab)
train_act=pd.read_csv('../input/bigcontest2019/train_activity.csv')

test1_act=pd.read_csv('../input/bigcontest2019/test1_activity.csv')

test2_act=pd.read_csv('../input/bigcontest2019/test2_activity.csv')

print('train_act.shape :',train_act.shape)

print('test1_act.shape :',test1_act.shape)

print('test2_act.shape :',test2_act.shape)

train_act.head()
resumetable(train_act)
# train_act['Train_or_Test']="Train"

# test1_act['Train_or_Test']="Test1"

# test2_act['Train_or_Test']="Test2"

# all_act=pd.concat([train_act,test1_act,test2_act],axis=0)

# train_act.drop('Train_or_Test',axis=1,inplace=True)

# test1_act.drop('Train_or_Test',axis=1,inplace=True)

# test2_act.drop('Train_or_Test',axis=1,inplace=True)

# all_act.head()
# all_act.tail()
train_com=pd.read_csv('../input/bigcontest2019/train_combat.csv')

test1_com=pd.read_csv('../input/bigcontest2019/test1_combat.csv')

test2_com=pd.read_csv('../input/bigcontest2019/test2_combat.csv')

print('train_com.shape :',train_com.shape)

print('test1_com.shape :',test1_com.shape)

print('test2_com.shape :',test2_com.shape)

train_com.head()
resumetable(train_com)
# train_com['Train_or_Test']="Train"

# test1_com['Train_or_Test']="Test1"

# test2_com['Train_or_Test']="Test2"

# all_com=pd.concat([train_com,test1_com,test2_com],axis=0)

# train_com.drop('Train_or_Test',axis=1,inplace=True)

# test1_com.drop('Train_or_Test',axis=1,inplace=True)

# test2_com.drop('Train_or_Test',axis=1,inplace=True)

# all_com.head()
# all_com.tail()
train_ple=pd.read_csv('../input/bigcontest2019/train_pledge.csv')

test1_ple=pd.read_csv('../input/bigcontest2019/test1_pledge.csv')

test2_ple=pd.read_csv('../input/bigcontest2019/test2_pledge.csv')

print('train_ple.shape :',train_ple.shape)

print('test1_ple.shape :',test1_ple.shape)

print('test2_ple.shape :',test2_ple.shape)

train_ple.head()
resumetable(train_ple)
# train_ple['Train_or_Test']="Train"

# test1_ple['Train_or_Test']="Test1"

# test2_ple['Train_or_Test']="Test2"

# all_ple=pd.concat([train_ple,test1_ple,test2_ple],axis=0)

# train_ple.drop('Train_or_Test',axis=1,inplace=True)

# test1_ple.drop('Train_or_Test',axis=1,inplace=True)

# test2_ple.drop('Train_or_Test',axis=1,inplace=True)

# all_ple.head()
# all_ple.tail()
train_tra=pd.read_csv('../input/bigcontest2019/train_trade.csv')

test1_tra=pd.read_csv('../input/bigcontest2019/test1_trade.csv')

test2_tra=pd.read_csv('../input/bigcontest2019/test2_trade.csv')

print('train_tra.shape :',train_tra.shape)

print('test1_tra.shape :',test1_tra.shape)

print('test2_tra.shape :',test2_tra.shape)

train_tra.head()
resumetable(train_tra)
# train_tra['Train_or_Test']="Train"

# test1_tra['Train_or_Test']="Test1"

# test2_tra['Train_or_Test']="Test2"

# all_tra=pd.concat([train_tra,test1_tra,test2_tra],axis=0)

# train_tra.drop('Train_or_Test',axis=1,inplace=True)

# test1_tra.drop('Train_or_Test',axis=1,inplace=True)

# test2_tra.drop('Train_or_Test',axis=1,inplace=True)

# all_tra.head()
# all_tra.tail()
train_pay=pd.read_csv('../input/bigcontest2019/train_payment.csv')

test1_pay=pd.read_csv('../input/bigcontest2019/test1_payment.csv')

test2_pay=pd.read_csv('../input/bigcontest2019/test2_payment.csv')

print('train_pay.shape :',train_pay.shape)

print('test1_pay.shape :',test1_pay.shape)

print('test2_pay.shape :',test2_pay.shape)

train_pay.head()
resumetable(train_pay)
# train_pay['Train_or_Test']="Train"

# test1_pay['Train_or_Test']="Test1"

# test2_pay['Train_or_Test']="Test2"

# all_pay=pd.concat([train_pay,test1_pay,test2_pay],axis=0)

# train_pay.drop('Train_or_Test',axis=1,inplace=True)

# test1_pay.drop('Train_or_Test',axis=1,inplace=True)

# test2_pay.drop('Train_or_Test',axis=1,inplace=True)

# all_pay.head()
# all_pay.tail()
train_valid=pd.read_csv('../input/bigcontest2019/train_valid_user_id.csv')

train_valid.head()
train_act.head()
train_act.groupby(['day','acc_id','char_id','server']).count().shape
train_act.shape
train_act[['acc_id','char_id','server']].groupby(['acc_id','char_id']).nunique()[train_act[['acc_id','char_id','server']].groupby(['acc_id','char_id']).nunique()['server']>1].shape
train_act[['acc_id','char_id','server']].groupby(['acc_id','char_id']).nunique()['server'].max()
print('동일 캐릭터로 다른 서버가 7개 기록된 경우의 수 :',train_act[['acc_id','char_id','server']].groupby(['acc_id','char_id']).nunique()[train_act[['acc_id','char_id','server']].groupby(['acc_id','char_id']).nunique()['server']==7].shape[0])
train_act[['acc_id','char_id','server']].groupby(['acc_id','char_id']).nunique()[train_act[['acc_id','char_id','server']].groupby(['acc_id','char_id']).nunique()['server']==7].head()
train_act[(train_act['acc_id']==38)&(train_act['char_id']==67497)]
plt.figure(figsize=(15,7))

s1=sns.countplot(train_act['server'])

s1.set(title='Server count')
print('bi 서버 기록된 개수 :',train_act[train_act['server']=='bi'].shape[0])

train_act[train_act['server']=='bi']
train_act[(train_act['playtime']==0)].shape
train_act[(train_act['playtime']==0)&(train_act['fishing']>0)].shape

# playtime이 0일 때 fishing이 0이 아닌 경우는 7145
train_act[(train_act['playtime']==0)&(train_act['fishing']>0)].head()
train_act[(train_act['playtime']==0)&(train_act['fishing']==0)]
train_act[(train_act['acc_id']==94800) & (train_act['char_id']==42512)]
train_act['fishing'].max()
train_lab[train_lab['acc_id']==94800]
train_act[(train_act['acc_id']==88486) & (train_act['char_id']==350129)]
train_act['private_shop'].max()
train_act[train_act['fishing']==train_act['fishing'].max()].shape
train_act[(train_act['fishing']==train_act['fishing'].max()) & (train_act['playtime']==0)].shape
train_act[(train_act['fishing']==train_act['fishing'].max()) & (train_act['playtime']>0)]
print('fishing==0 :',train_act[train_act['fishing']==0].shape[0])

print('train_activity 데이터 개수 :',train_act.shape[0])

print('activity데이터에서 fishing이 0이 아닌 비율 :',train_act[train_act['fishing']!=0].shape[0]/train_act.shape[0])
train_act[train_act['death']>0].head()
print('death가 0이 아니고, npc_kill, solo_exp, party_exp, quest_exp, rich_monster가 0인 개수 : ',train_act[(train_act['death']>0)&(train_act['npc_kill']==0)&(train_act['solo_exp']==0)&(train_act['party_exp']==0)&(train_act['quest_exp']==0)&(train_act['rich_monster']==0)].shape[0])
train_act[(train_act['death']>0)&(train_act['npc_kill']==0)&(train_act['solo_exp']==0)&(train_act['party_exp']==0)&(train_act['quest_exp']==0)&(train_act['rich_monster']==0)].head(10)
train_act.head()
label=train_lab.copy()

label['is_survival']=label['survival_time'].map(lambda x: 1 if x==64 else 0)

label.head()
act_lab=pd.merge(train_act,label,how='left',on='acc_id')

act_lab.head()
act_lab.isna().sum()
fig, ax = plt.subplots(4,4,figsize=(20,20))

l1=['playtime','npc_kill','solo_exp','party_exp','quest_exp','rich_monster','death','revive','exp_recovery',

    'fishing','private_shop','game_money_change','enchant_count']

for i in range(13):

    sns.boxplot(x='is_survival',y=l1[i],data=act_lab,ax=ax[i//4,i%4])

    ax[i//4,i%4].set(title = str(l1[i]) + 'by is_survival')
fig, ax = plt.subplots(4,3,figsize=(20,20))

l1=['playtime','npc_kill','solo_exp','party_exp','quest_exp','death','revive','exp_recovery',

    'fishing','private_shop','game_money_change','enchant_count']

for i in range(12):

    sns.kdeplot(act_lab.loc[act_lab['is_survival']==1,l1[i]],label='survive',ax=ax[i//3,i%3])

    sns.kdeplot(act_lab.loc[act_lab['is_survival']==0,l1[i]],label='leave',ax=ax[i//3,i%3])

    ax[i//3,i%3].set(title = str(l1[i]) + ' by is_survival')
t1=len(act_lab[act_lab['is_survival']==0])

t2=len(act_lab[act_lab['is_survival']==1])

fig, ax = plt.subplots(1,2,figsize=(15,7))

sns.countplot(act_lab.loc[act_lab['is_survival']==0,'rich_monster'],ax=ax[0])

sns.countplot(act_lab.loc[act_lab['is_survival']==1,'rich_monster'],ax=ax[1])

ax[0].set(title='rich_monster attacked by leave',ylim=(0,900000))

ax[1].set(title='rich_monster attacked by survive',ylim=(0,900000))

for p in ax[0].patches:

    height = p.get_height()

    ax[0].text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}%'.format(height/t1*100),

           ha = 'center')

for p in ax[1].patches:

    height = p.get_height()

    ax[1].text(p.get_x() + p.get_width()/2.,

               height + 3,

               '{:2.2f}%'.format(height/t2*100),

               ha = 'center')
train_act.columns
print('캐릭터 개수.max() :',train_act[['acc_id','char_id']].groupby(['acc_id'])['char_id'].nunique().max())

char_count = train_act[['acc_id','char_id']].groupby('acc_id')['char_id'].nunique().reset_index(name='char_count')
day_count = train_act[['acc_id','day']].groupby('acc_id')['day'].nunique().reset_index(name='day_count')
print('서버 개수.max() :',train_act[['acc_id','server']].groupby(['acc_id'])['server'].nunique().max())

server_count = train_act[['acc_id','server']].groupby('acc_id')['server'].nunique().reset_index(name='server_count')
playtime_sum = train_act[['acc_id','playtime']].groupby('acc_id')['playtime'].sum().reset_index(name='playtime_sum')

playtime_mean = train_act[['acc_id','playtime']].groupby('acc_id')['playtime'].mean().reset_index(name='playtime_mean')

playtime_max = train_act[['acc_id','playtime']].groupby('acc_id')['playtime'].max().reset_index(name='playtime_max')

playtime_smm = pd.merge(playtime_sum, playtime_mean, on='acc_id', how='left')

playtime_smm = pd.merge(playtime_smm, playtime_max, on='acc_id', how='left')

playtime_smm.head()
npc_kill_sum = train_act[['acc_id','npc_kill']].groupby('acc_id')['npc_kill'].sum().reset_index(name='npc_kill_sum')

npc_kill_mean = train_act[['acc_id','npc_kill']].groupby('acc_id')['npc_kill'].mean().reset_index(name='npc_kill_mean')

npc_kill_max = train_act[['acc_id','npc_kill']].groupby('acc_id')['npc_kill'].max().reset_index(name='npc_kill_max')

npc_kill_smm = pd.merge(npc_kill_sum, npc_kill_mean, on='acc_id', how='left')

npc_kill_smm = pd.merge(npc_kill_smm, npc_kill_max, on='acc_id', how='left')

npc_kill_smm.head()
solo_sum = train_act[['acc_id','solo_exp']].groupby('acc_id')['solo_exp'].sum().reset_index(name='solo_exp_sum')

solo_mean = train_act[['acc_id','solo_exp']].groupby('acc_id')['solo_exp'].mean().reset_index(name='solo_exp_mean')

solo_max = train_act[['acc_id','solo_exp']].groupby('acc_id')['solo_exp'].max().reset_index(name='solo_exp_max')

solo_smm = pd.merge(solo_sum, solo_mean, on='acc_id', how='left')

solo_smm = pd.merge(solo_smm, solo_max, on='acc_id', how='left')

solo_smm.head()
party_sum = train_act[['acc_id','party_exp']].groupby('acc_id')['party_exp'].sum().reset_index(name='party_exp_sum')

party_mean = train_act[['acc_id','party_exp']].groupby('acc_id')['party_exp'].mean().reset_index(name='party_exp_mean')

party_max = train_act[['acc_id','party_exp']].groupby('acc_id')['party_exp'].max().reset_index(name='party_exp_max')

party_smm = pd.merge(party_sum, party_mean, on='acc_id', how='left')

party_smm = pd.merge(party_smm, party_max, on='acc_id', how='left')

party_smm.head()
quest_sum = train_act[['acc_id','quest_exp']].groupby('acc_id')['quest_exp'].sum().reset_index(name='quest_exp_sum')

quest_mean = train_act[['acc_id','quest_exp']].groupby('acc_id')['quest_exp'].mean().reset_index(name='quest_exp_mean')

quest_max = train_act[['acc_id','quest_exp']].groupby('acc_id')['quest_exp'].max().reset_index(name='quest_exp_max')

quest_smm = pd.merge(quest_sum, quest_mean, on='acc_id', how='left')

quest_smm = pd.merge(quest_smm, quest_max, on='acc_id', how='left')

quest_smm.head()
rich_sum = train_act[['acc_id','rich_monster']].groupby('acc_id')['rich_monster'].sum().reset_index(name='rich_sum')

rich_mean = train_act[['acc_id','rich_monster']].groupby('acc_id')['rich_monster'].mean().reset_index(name='rich_mean')

rich_att = train_act[['acc_id','rich_monster']].groupby('acc_id')['rich_monster'].max().reset_index(name='rich_attacked')

rich_smm = pd.merge(rich_sum, rich_mean, on='acc_id', how='left')

rich_smm = pd.merge(rich_smm, rich_att, on='acc_id', how='left')

rich_smm.head()
death_sum = train_act[['acc_id','death']].groupby('acc_id')['death'].sum().reset_index(name='death_sum')

death_mean = train_act[['acc_id','death']].groupby('acc_id')['death'].mean().reset_index(name='death_mean')

death_max = train_act[['acc_id','death']].groupby('acc_id')['death'].max().reset_index(name='death_max')

death_smm = pd.merge(death_sum, death_mean, on='acc_id', how='left')

death_smm = pd.merge(death_smm, death_max, on='acc_id', how='left')

death_smm.head()
revive_sum = train_act[['acc_id','revive']].groupby('acc_id')['revive'].sum().reset_index(name='revive_sum')

revive_mean = train_act[['acc_id','revive']].groupby('acc_id')['revive'].mean().reset_index(name='revive_mean')

revive_max = train_act[['acc_id','revive']].groupby('acc_id')['revive'].max().reset_index(name='revive_max')

revive_smm = pd.merge(revive_sum, revive_mean, on='acc_id', how='left')

revive_smm = pd.merge(revive_smm, revive_max, on='acc_id', how='left')

revive_smm.head()
exp_recovery_sum = train_act[['acc_id','exp_recovery']].groupby('acc_id')['exp_recovery'].sum().reset_index(name='exp_recovery_sum')

exp_recovery_mean = train_act[['acc_id','exp_recovery']].groupby('acc_id')['exp_recovery'].mean().reset_index(name='exp_recovery_mean')

exp_recovery_max = train_act[['acc_id','exp_recovery']].groupby('acc_id')['exp_recovery'].max().reset_index(name='exp_recovery_max')

exp_recovery_smm = pd.merge(exp_recovery_sum, exp_recovery_mean, on='acc_id', how='left')

exp_recovery_smm = pd.merge(exp_recovery_smm, exp_recovery_max, on='acc_id', how='left')

exp_recovery_smm.head()
fishing_sum = train_act[['acc_id','fishing']].groupby('acc_id')['fishing'].sum().reset_index(name='fishing_sum')

fishing_mean = train_act[['acc_id','fishing']].groupby('acc_id')['fishing'].mean().reset_index(name='fishing_mean')

fishing_max = train_act[['acc_id','fishing']].groupby('acc_id')['fishing'].max().reset_index(name='fishing_max')

fishing_smm = pd.merge(fishing_sum, fishing_mean, on='acc_id', how='left')

fishing_smm = pd.merge(fishing_smm, fishing_max, on='acc_id', how='left')

fishing_smm.head()
private_sum = train_act[['acc_id','private_shop']].groupby('acc_id')['private_shop'].sum().reset_index(name='private_shop_sum')

private_mean = train_act[['acc_id','private_shop']].groupby('acc_id')['private_shop'].mean().reset_index(name='private_shop_mean')

private_max = train_act[['acc_id','private_shop']].groupby('acc_id')['private_shop'].max().reset_index(name='private_shop_max')

private_smm = pd.merge(private_sum, private_mean, on='acc_id', how='left')

private_smm = pd.merge(private_smm, private_max, on='acc_id', how='left')

private_smm.head()
game_money_sum = train_act[['acc_id','game_money_change']].groupby('acc_id')['game_money_change'].sum().reset_index(name='game_money_sum')

game_money_mean = train_act[['acc_id','game_money_change']].groupby('acc_id')['game_money_change'].mean().reset_index(name='game_money_mean')

game_money_max = train_act[['acc_id','game_money_change']].groupby('acc_id')['game_money_change'].max().reset_index(name='game_money_max')

game_money_min = train_act[['acc_id','game_money_change']].groupby('acc_id')['game_money_change'].min().reset_index(name='game_money_min')

game_money_smm = pd.merge(game_money_sum, game_money_mean, on='acc_id', how='left')

game_money_smm = pd.merge(game_money_smm, game_money_max, on='acc_id', how='left')

game_money_smm = pd.merge(game_money_smm, game_money_min, on='acc_id', how='left')

game_money_smm.head()
enchant_sum = train_act[['acc_id','enchant_count']].groupby('acc_id')['enchant_count'].sum().reset_index(name='enchant_count_sum')

enchant_sum['enchant_count_sum'] = enchant_sum['enchant_count_sum'].map(lambda x: 1 if x>0 else 0)

enchant_sum.head()
train_act.columns
holy_label = pd.merge(train_lab, train_valid, on='acc_id', how='left')

holy_label = pd.merge(holy_label, char_count, on='acc_id', how='left')

holy_label = pd.merge(holy_label, day_count, on='acc_id', how='left')

holy_label = pd.merge(holy_label, server_count, on='acc_id', how='left')

holy_label = pd.merge(holy_label, playtime_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, npc_kill_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, solo_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, party_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, quest_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, rich_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, death_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, revive_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, exp_recovery_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, fishing_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, private_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, game_money_smm, on='acc_id', how='left')

holy_label = pd.merge(holy_label, enchant_sum, on='acc_id', how='left')

holy_label.head()
print(holy_label.shape)

holy_label.isna().sum()
train = holy_label[holy_label['set']=='Train']

test = holy_label[holy_label['set']=='Validation']

train.head()
test.head()
x_train = train.drop(['acc_id','survival_time','amount_spent','set'],axis=1)

y1_train = train['survival_time']

y2_train = train['amount_spent']

x_test = test.drop(['acc_id','survival_time','amount_spent','set'],axis=1)
lgb_params = {'learning_rate':0.1,

              'max_depth':6,

              'boosting':'gbdt',

              'objective':'regression',

              'metric':'rmse',

              'num_leaves':31,

              'min_child_samples':1,}
trainset = lgb.Dataset(x_train, y1_train)



cv_output = lgb.cv(lgb_params,

                   trainset,

                   num_boost_round = 10000,

                   nfold = 5,

                   early_stopping_rounds = 200,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds1 = np.array(list(cv_output.values())).shape[1]

print('best_rounds :', best_rounds1)
lgb_model = lgb.train(lgb_params,

                      trainset,

                      num_boost_round = best_rounds1)

y1_pred = lgb_model.predict(x_test)

y1_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

lgb.plot_importance(lgb_model,ax=ax)

plt.show()
xgb_params={'eta':0.01,

            'max_depth':6,

            'objective':'reg:squarederror',

            'metric':'rmse',

            'min_child_samples':100}
dtrain = xgb.DMatrix(x_train,y2_train)

dtest = xgb.DMatrix(x_test)



cv_output = xgb.cv(xgb_params,

                   dtrain,

                   num_boost_round = 10000,

                   nfold = 5,

                   early_stopping_rounds = 200,

                   verbose_eval = 100,

                   show_stdv = False)



best_rounds2 = cv_output.index.size



print('Best rounds :',best_rounds2)
xgb_model = xgb.train(xgb_params,

                      dtrain,

                      num_boost_round = best_rounds2)

y2_pred = xgb_model.predict(dtest)

y2_pred[:20]
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(xgb_model,ax=ax)

plt.show()
pred = pd.DataFrame({'acc_id':test['acc_id'],'survival_time':y1_pred,'amount_spent':y2_pred})

pred.head()
def ss(x):

    if x>64:

        return 64

    elif x<1:

        return 1

    else:

        return x
pred['survival_time'] = pred['survival_time'].map(ss)

pred['survival_time'] = np.round(pred['survival_time'])

pred.head()
# ------------------------------------------------------------------------ #

# INPUT

#    predicted_label : 예측 답안지 파일 경로 

#    actual_label    : 실제 답안지 파일 경로

#

# OUTPUT             : 유저 기대이익 총합 

#

#

# - 예측 답안지를 실제 답안과 비교하여 유저 기대이익 총합을 계산하는 함수

# - 함수의 계산방식은 문제 설명서에 기술된 기대이익 산출식과 동일

# ------------------------------------------------------------------------ #



#필요한 모듈 import

import pandas as pd

import numpy as np

import sys



def score_function(predict_label, actual_label):

    

    predict = predict_label # 예측 답안 파일 불러오기

    actual = actual_label # 실제 답안 파일 불러오기 



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

        print(score)

    return score
actual_label = test[['acc_id','survival_time','amount_spent']]
score = score_function(pred, actual_label)
pred.to_csv('pred.csv',index=False)