import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('bmh')

import matplotlib as mpl

import os

print(os.listdir("../input/"))

import gc



import warnings 

warnings.filterwarnings('ignore')

def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Min'] = df.min().values

    summary['Max'] = df.max().values

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values



    return summary
train_label = pd.read_csv('../input/train_label.csv')



resumetable(train_label)
train_label.head()
train_act = pd.read_csv('../input/train_activity.csv')

test1_act = pd.read_csv('../input/test1_activity.csv')

test2_act = pd.read_csv('../input/test2_activity.csv')



print("train_activity shape: {}".format(train_act.shape))

print("test1_activity shape: {}".format(test1_act.shape))

print("test2_activity shape: {}".format(test2_act.shape))



train_act['Train_or_Test'] = "Train"

test1_act['Train_or_Test'] = "Test1"

test2_act['Train_or_Test'] = "Test2"



all_act = pd.concat([train_act,test1_act,test2_act],axis= 0)
resumetable(all_act)
all_act.head()
train_trade = pd.read_csv('../input/train_trade.csv')

test1_trade = pd.read_csv('../input/test1_trade.csv')

test2_trade = pd.read_csv('../input/test2_trade.csv')



print("train_trade shape: {}".format(train_trade.shape))

print("test1_trade shape: {}".format(test1_trade.shape))

print("test2_trade shape: {}".format(test2_trade.shape))



train_trade['Train_or_Test'] = "Train"

test1_trade['Train_or_Test'] = "Test1"

test2_trade['Train_or_Test'] = "Test2"



all_trade = pd.concat([train_trade, test1_trade, test2_trade], axis=0)
resumetable(all_trade)
all_trade.head()
train_combat = pd.read_csv('../input/train_combat.csv')

test1_combat = pd.read_csv('../input/test1_combat.csv')

test2_combat = pd.read_csv('../input/test2_combat.csv')



print("train_combat.csv: {}".format(train_combat.shape))

print("test1_combat.csv: {}".format(test1_combat.shape))

print("test2_combat.csv: {}".format(test2_combat.shape))



train_combat['Train_or_Test'] = "Train"

test1_combat['Train_or_Test'] = 'Test1'

test2_combat['Train_or_Test'] = 'Test2'



all_combat = pd.concat([train_combat,test1_combat,test2_combat])
resumetable(all_combat)
all_combat.head()
train_pledge = pd.read_csv('../input/train_pledge.csv')

test1_pledge = pd.read_csv('../input/test1_pledge.csv')

test2_pledge = pd.read_csv('../input/test2_pledge.csv')



print("train_pledge shape: {}".format(train_pledge.shape))

print("test1_pledge shape: {}".format(test1_pledge.shape))

print("test2_pledge shape: {}".format(test2_pledge.shape))



train_pledge['Train_or_Test'] = 'Train'

test1_pledge['Train_or_Test'] = 'Test1'

test2_pledge['Train_or_Test'] = 'Test2'



all_pledge = pd.concat([train_pledge,test1_pledge,test2_pledge])
resumetable(all_pledge)
all_pledge.head()
train_pay = pd.read_csv('../input/train_payment.csv')

test1_pay = pd.read_csv('../input/test1_payment.csv')

test2_pay = pd.read_csv('../input/test2_payment.csv')



print("train payment shape : {}".format(train_pay.shape))

print("test1 payment shape : {}".format(test1_pay.shape))

print("test2 payment shape : {}".format(test2_pay.shape))



train_pay['Train_or_Test'] = 'Train'

test1_pay['Train_or_Test'] = 'Test1'

test2_pay['Train_or_Test'] = 'Test2'



all_pay = pd.concat([train_pay,test1_pay,test2_pay])
resumetable(all_pay)
train_label_copy = train_label.copy()

train_label_copy['isSurvival'] = train_label_copy['survival_time'].map(lambda x: 1 if x == 64 else 0)
total = len(train_label_copy)

ax = sns.countplot(train_label_copy.isSurvival)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot \n # 0: Churn | 1: Retention")
f, ax = plt.subplots(1,2, figsize =(12,5))

sns.distplot(train_label.survival_time,ax = ax[0])

ax[0].set_title("Survival Time Density plot")



sns.distplot(train_label.amount_spent,ax = ax[1])

ax[1].set_title("Amt Spent Density plot")
plt.figure(figsize=(12,5))

sns.scatterplot(x = train_label.survival_time, y = train_label.amount_spent)

plt.title("Surivival Time and Amt Spent Scatter plot")
train_act_copy = train_act.copy()

train_act_copy.drop(['day','acc_id','char_id','server','boss_monster'],inplace = True, axis = 1)



print("Activity Data Quantile")

print(train_act_copy.quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
f, ax = plt.subplots(4,3, figsize =(24,10))

sns.kdeplot(train_act.playtime,shade = True, ax = ax[0,0])

ax[0,0].set_title("Daily play time Density plot")



sns.kdeplot(train_act.npc_kill,shade = True, ax = ax[0,1])

ax[0,1].set_title("The number of killing about NPC Density plot")



sns.kdeplot(train_act.solo_exp,shade = True, ax = ax[0,2])

ax[0,2].set_title("Solo play experience Density plot")



sns.kdeplot(train_act.party_exp,shade = True, ax = ax[1,0])

ax[1,0].set_title("Party play experience Density plot")



sns.kdeplot(train_act.quest_exp,shade = True, ax = ax[1,1])

ax[1,1].set_title("Quest experience Density plot")



sns.kdeplot(train_act.death,shade = True, ax = ax[1,2])

ax[1,2].set_title("Death Density plot")



sns.kdeplot(train_act.revive,shade = True, ax = ax[2,0])

ax[2,0].set_title("Revive Density plot")



sns.kdeplot(train_act.exp_recovery,shade = True, ax = ax[2,1])

ax[2,1].set_title("Restore experience Density plot")



sns.kdeplot(train_act.fishing,shade = True, ax = ax[2,2])

ax[2,2].set_title("Daily Fishing Time Density plot")



sns.kdeplot(train_act.private_shop,shade = True, ax = ax[3,0])

ax[3,0].set_title("Daily Personal store Time Density plot")



sns.kdeplot(train_act.game_money_change,shade = True, ax = ax[3,1])

ax[3,1].set_title("Daily Adena variation Density plot")



sns.kdeplot(train_act.enchant_count,shade = True, ax = ax[3,2])

ax[3,2].set_title("Try Enchantment over 7 level item  Density plot")



plt.subplots_adjust(hspace = .6, top = .9)

plt.show()
cat_colname = {'day','acc_id','char_id','server','Train_or_Test','boss_monster'}

act_colname = list(train_act.columns)

act_colname = [e for e in act_colname if e not in cat_colname]
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(train_act[act_colname].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.title('Correlation Plot of Activity Data', fontsize = 18)
total = len(train_act)

ax = sns.countplot(train_act.boss_monster)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Boss monster attack Count plot \n # 0: NO | 1: YES #")
def day_boxplot(dt, var, subplot_x, subplot_y, figsize_x, figsize_y):

    plt.figure(figsize = (20,35))

    

    for i,j in enumerate(var):

        plt.subplot(subplot_x,subplot_y,i+1)

        

        sns.boxplot(x = 'day', y = '{}'.format(j) , data = dt )

        plt.title('Boxplot for {} by Day'.format(j))

        plt.xlabel('Day')
day_boxplot(dt = train_act, var = act_colname,subplot_x = 6, subplot_y = 2, figsize_x = 20, figsize_y = 35  )
print("Item Amount & Item Price Quntile")

print(train_trade[['item_amount','item_price']].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
f,ax = plt.subplots(1,2,figsize = (12,5))



sns.kdeplot(train_trade['item_amount'],ax= ax[0], shade = True)

ax[0].set_title('Distribution by Item Amount in Trade')



sns.kdeplot(train_trade['item_price'], ax = ax[1], shade = True)

ax[1].set_title('Distribution by Item Price in Trade')
f, ax = plt.subplots(2,1,figsize = (10,10))



sns.countplot(x = 'day', hue = 'type', data = train_trade,ax = ax[0])

ax[0].set_title('Countplot by Daily Trade Occurance',fontsize = 16)

ax[0].set_xlabel('Day')

ax[0].set_ylabel('Count')



sns.countplot(x = 'server', hue = 'type', data = train_trade,ax = ax[1])

ax[1].set_title('Countplot by Trade Occurance by Server',fontsize = 16)

ax[1].set_xlabel('Server')

ax[1].set_ylabel('Count')



plt.subplots_adjust(hspace = 0.4)
tmp = train_trade.copy()

tmp = tmp[['item_type','item_price']].groupby(['item_type']).median()

tmp = tmp.reset_index()



plt.figure(figsize = (16,10))

plt.suptitle('Item type Distribution',fontsize = 22)



plt.subplot(222)

g = sns.countplot(x = 'item_type', data = train_trade)

gt = g.twinx()

gt = sns.pointplot( x = 'item_type',y = 'item_price',data = tmp,

                  color = 'black', legend = False,

                  order = ['etc','adena', 'accessory','armor','spell','enchant_scroll','weapon'])

gt.set_ylabel('Item Price',fontsize = 16)

g.set_title('Item Type Distribution', fontsize = 19)

g.set_xlabel('Item Type Name', fontsize = 17)

g.set_ylabel('Count',fontsize = 17)



plt.subplot(221)

g2 = sns.countplot(x = 'item_type', hue = 'type', data = train_trade)

g2.set_title('Item Type Distribution by Trade Type',fontsize = 19)

g2.set_xlabel('Item Type',fontsize = 17)

g2.set_ylabel('Count',fontsize = 17)



plt.subplot(212)

g3 = sns.boxenplot(x = 'item_type', y = 'item_price', data = train_trade)

g3.set_title('Item Price by Type', fontsize = 20)

g3.set_xlabel('Item Type Names',fontsize = 17)

g3.set_ylabel('Item Price')



plt.subplots_adjust(hspace = 0.8,top = 0.85)
print('Combat Data Quantile')

print(train_combat[['pledge_cnt','random_attacker_cnt','random_defender_cnt',

                   'temp_cnt','same_pledge_cnt','etc_cnt','num_opponent']].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
plt.figure(figsize=(24,10))



plt.subplot(241)

g1 = sns.kdeplot(train_combat.pledge_cnt, shade = True)

g1.set_title('Number of Pledge Battle Fights')



plt.subplot(242)

g2 = sns.kdeplot(train_combat.random_attacker_cnt, shade = True)

g2.set_title('Number of Random Attack')



plt.subplot(243)

g3 = sns.kdeplot(train_combat.random_defender_cnt, shade = True)

g3.set_title('Number of Random Defence')



plt.subplot(244)

g4 = sns.kdeplot(train_combat.temp_cnt, shade = True)

g4.set_title('Number of Temporary Battle')



plt.subplot(245)

g5 = sns.kdeplot(train_combat.same_pledge_cnt, shade = True)

g5.set_title('Number of Same Pledge Battle')



plt.subplot(246)

g6 = sns.kdeplot(train_combat.etc_cnt, shade = True)

g6.set_title('Number of etc. Battle')



plt.subplot(247)

g7 = sns.kdeplot(train_combat.num_opponent, shade = True)

g7.set_title('Number of Opponent Character to  Battle')



plt.show()
resumetable(train_combat.loc[train_combat.num_opponent == 0])
cat_colname = {'day','acc_id','char_id','server','Train_or_Test','boss_monster','class','level'}

combat_colname = list(train_combat.columns)

combat_colname = [e for e in combat_colname if e not in cat_colname]
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(train_combat[combat_colname].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
tmp_class = train_combat.copy()

tmp_class = tmp_class[['class','etc_cnt']].groupby(['class']).mean()

tmp_class = tmp_class.reset_index()



tmp_level = train_combat.copy()

tmp_level = tmp_level[['level','etc_cnt']].groupby(['level']).mean()

tmp_level = tmp_level.reset_index()



plt.figure(figsize = (15,8))



plt.subplot(221)

g1 = sns.countplot(x = 'class', data = train_combat)

gt = g1.twinx()

gt = sns.pointplot(x = 'class', y = 'etc_cnt',data = tmp_class,

                  order = list(range(0,8)), color = 'black', legend = False)

g1.set_title('Distribution by Class',fontsize = 17)

g1.set_xlabel('Class',fontsize = 16)

g1.set_ylabel('Count', fontsize = 16)

gt.set_ylabel('Mean of # of etc Battle',fontsize = 14)



plt.subplot(222)

g2 = sns.countplot(x = 'level', data = train_combat)

gt = g2.twinx()

gt = sns.pointplot(x = 'level', y = 'etc_cnt',data = tmp_level,

                  order = list(range(0,18)), color = 'black', legend = False)

g2.set_title('Distribution by Level',fontsize= 17)

g2.set_xlabel('Level',fontsize = 16)

g2.set_ylabel('Count', fontsize = 16)

gt.set_ylabel('Mean of # of etc Battle',fontsize = 14)



plt.subplots_adjust(wspace = 0.3)



plt.figure(figsize = (15,8))



plt.subplot(212)

g3 = sns.boxenplot(x = 'class', y = 'etc_cnt', data = train_combat)

g3.set_title("Class & Number of etc Battle Boxplot",fontsize = 16)







plt.figure(figsize = (15,8))



plt.subplot(212)

g4 = sns.boxenplot(x = 'level', y = 'etc_cnt', data = train_combat)

g4.set_title('Level & Number of etc Battle Boxplot',fontsize = 16)



cat_colname = {'day','acc_id','char_id','server','Train_or_Test','boss_monster','class','level'}

combat_colname = list(train_combat.columns)

combat_colname = [e for e in combat_colname if e not in cat_colname]
day_boxplot(dt = train_combat, var = combat_colname,subplot_x = 4, subplot_y = 2, figsize_x = 20, figsize_y = 35  )
tmp_train_pledge = train_pledge.drop_duplicates(subset = ['server','day','pledge_id'])



resumetable(tmp_train_pledge)
tmp_train_pledge[tmp_train_pledge.duplicated(['server','day','pledge_id'])]
print("Plegdge Data Quantile")

print(tmp_train_pledge[['play_char_cnt','combat_char_cnt','random_attacker_cnt','random_defender_cnt',

                   'temp_cnt','same_pledge_cnt','etc_cnt','combat_play_time','non_combat_play_time']].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
plt.figure(figsize = (24,15))

plt.subplot(4,3,1)

sns.kdeplot(tmp_train_pledge.play_char_cnt, shade = True)

plt.title('Number of Pledgers in the game')



plt.subplot(4,3,2)

sns.kdeplot(tmp_train_pledge.combat_char_cnt, shade = True)

plt.title('Number of Pledgers in the Battle')



plt.subplot(4,3,3)

sns.kdeplot(tmp_train_pledge.pledge_combat_cnt, shade = True)

plt.title("Sum of Number of Pledge Battles")



plt.subplot(4,3,4)

sns.kdeplot(tmp_train_pledge.random_attacker_cnt, shade = True)

plt.title("Number of Random Attack in The Pledger")



plt.subplot(4,3,5)

sns.kdeplot(tmp_train_pledge.random_defender_cnt, shade = True)

plt.title("Number of Random Defence in The Pledger")



plt.subplot(4,3,6)

sns.kdeplot(tmp_train_pledge.same_pledge_cnt, shade = True)

plt.title('Number of Same Pledge Battle in The Pledger')



plt.subplot(4,3,7)

sns.kdeplot(tmp_train_pledge.temp_cnt, shade = True)

plt.title('Number of Temporary Battle in The Pledger')



plt.subplot(4,3,8)

sns.kdeplot(tmp_train_pledge.etc_cnt, shade = True)

plt.title('Number of etc. Battle in The Pledger')



plt.subplot(4,3,9)

sns.kdeplot(tmp_train_pledge.combat_play_time, shade = True)

plt.title('Sum of The Battle Pledger Play Time')



plt.subplot(4,3,10)

sns.kdeplot(tmp_train_pledge.non_combat_play_time, shade = True)

plt.title('Sum of The Non - Battle Pledger Play Time')
cat_colname = {'day','acc_id','char_id','server','Train_or_Test','pledge_id'}

pledge_colname = list(train_pledge.columns)

pledge_colname = [e for e in pledge_colname if e not in cat_colname]
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(tmp_train_pledge[pledge_colname].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.title('Correlation of Pledge Data')
print("# of Unique value : ",all_pledge.loc[all_pledge['combat_play_time'] > 0].non_combat_play_time.nunique())

print("Unique value : ", all_pledge.loc[all_pledge['combat_play_time'] > 0].non_combat_play_time.unique())
condition1 = train_pledge.loc[~(train_pledge['combat_play_time'] != 0) & ~(train_pledge['non_combat_play_time'] == 0)]



sns.kdeplot(condition1.non_combat_play_time, shade = True)

plt.title('Sum of Non Battle Pledger Play Time')
condition1.head()
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(condition1[pledge_colname].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
print('Payment Quntile')

print(train_pay[['amount_spent']].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
sns.kdeplot(train_pay.amount_spent, shade = True)

plt.title('Amount of Payment')
plt.figure(figsize = (10,8))

sns.boxplot(x = 'day', y = 'amount_spent', data = train_pay)