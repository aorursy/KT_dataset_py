# preprocessing

import numpy as np

import pandas as pd 

import networkx



# graph

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



# model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



# evaluation

from sklearn.metrics import mean_squared_error



# utils

import warnings

warnings.filterwarnings('ignore')



import os

os.listdir('/kaggle/input/bigcontest2019/')
seed = 223
train_act = pd.read_csv('/kaggle/input/bigcontest2019/train_activity.csv')

train_pay = pd.read_csv('/kaggle/input/bigcontest2019/train_payment.csv')

train_pledge = pd.read_csv('/kaggle/input/bigcontest2019/train_pledge.csv')

train_trade = pd.read_csv('/kaggle/input/bigcontest2019/train_trade.csv')

train_combat = pd.read_csv('/kaggle/input/bigcontest2019/train_combat.csv')

print('train activity shape: ',train_act.shape)

print('train payment shape: ',train_pay.shape)

print('train pledge shape: ',train_pledge.shape)

print('train trade shape: ',train_trade.shape)

print('train combat shape: ',train_combat.shape)
train_label = pd.read_csv('/kaggle/input/bigcontest2019/train_label.csv')

print('train_label shape: ',train_label.shape)
train_label['survival_week'] = train_label.survival_time // 7

train_act_label = pd.merge(train_act, train_label, on='acc_id', how='left')
day_week = train_act_label[['acc_id','day','survival_week']].drop_duplicates()



plt.figure(figsize=(15,10))

ax1 = plt.subplot2grid((2,4), (0,0), colspan=2)

ax2 = plt.subplot2grid((2,4), (0,2), colspan=2)

ax3 = plt.subplot2grid((2,4), (1,0), colspan=4)

for i in range(10):

    day_df = day_week[day_week.survival_week==i]

    day_df = (day_df.day.value_counts() / day_df.shape[0]).reset_index()

    sns.lineplot(x='index', y='day', data=day_df, label='{} week'.format(i), ax=ax1)

ax1.set_title('Activity day distribution by Churn week')

ax1.set_xlabel('Day')

ax1.set_ylabel('Count')



for i in range(1,10):

    day_df = day_week[day_week.survival_week==i]

    day_df = (day_df.day.value_counts() / day_df.shape[0]).reset_index()

    sns.lineplot(x='index', y='day', data=day_df, label='{} week'.format(i), ax=ax2)

ax2.set_title('Activity day distribution by Churn week without 0 survival_week')

ax2.set_xlabel('Day')

ax2.set_ylabel('Count')





sns.lineplot(x='day', y='survival_week', data=day_week, ax=ax3)

ax3.set_title('Churn week distribution by activity day')

ax3.set_xlabel('Day')

ax3.set_ylabel('Survival week')

plt.show()
def feature_by_day(feature, data, label_df, title, y_label, method='sum'):

    if method=='sum':

        f_by_day = data.groupby(['acc_id','day'])[feature].sum().reset_index()

    elif method=='mean':

        f_by_day = data.groupby(['acc_id','day'])[feature].mean().reset_index()

    elif method=='median':

        f_by_day = data.groupby(['acc_id','day'])[feature].median().reset_index()

    elif method=='std':

        f_by_day = data.groupby(['acc_id','day'])[feature].std().reset_index()

        

    f_by_day = pd.merge(f_by_day, label_df, on='acc_id', how='left')

    plt.figure(figsize=(15,5))

    sns.lineplot(x='day', y=feature, hue='survival_week', data=f_by_day, palette='Reds_r')

    plt.legend(bbox_to_anchor=(1.1, 1))

    plt.title(title)

    plt.ylabel(y_label)

    plt.show()
feature_by_day(feature='playtime',

               data=train_act,

               label_df=train_label,

               title='Play time change over day by survival week',

               y_label='Play Time')
feature_by_day(feature='npc_kill',

               data=train_act,

               label_df=train_label,

               title='NPC kill change over day by survival week',

               y_label='NPC Kill')
feature_by_day(feature='solo_exp',

               data=train_act,

               label_df=train_label,

               title='Solo Exp change over day by survival week',

               y_label='Solo Exp')
feature_by_day(feature='party_exp',

               data=train_act,

               label_df=train_label,

               title='Party Exp change over day by survival week',

               y_label='Party Exp')
feature_by_day(feature='quest_exp',

               data=train_act,

               label_df=train_label,

               title='Quest Exp change over day by survival week',

               y_label='Quest Exp')
feature_by_day(feature='rich_monster',

               data=train_act,

               label_df=train_label,

               title='Rich monster change over day by survival week',

               y_label='Rich monster')
feature_by_day(feature='death',

               data=train_act,

               label_df=train_label,

               title='Death change over day by survival week',

               y_label='Death')
feature_by_day(feature='revive',

               data=train_act,

               label_df=train_label,

               title='Revive change over day by survival week',

               y_label='Revive')
feature_by_day(feature='exp_recovery',

               data=train_act,

               label_df=train_label,

               title='Exp recovery change over day by survival week',

               y_label='Exp recovery')
feature_by_day(feature='fishing',

               data=train_act,

               label_df=train_label,

               title='Fishing change over day by survival week',

               y_label='Fishing')
feature_by_day(feature='private_shop',

               data=train_act,

               label_df=train_label,

               title='Private shop change over day by survival week',

               y_label='Private shop')
feature_by_day(feature='game_money_change',

               data=train_act,

               label_df=train_label,

               title='Game money change change over day by survival week',

               y_label='Game money change')
feature_by_day(feature='enchant_count',

               data=train_act,

               label_df=train_label,

               title='Enchant count change over day by survival week',

               y_label='Enchant count')
feature_by_day(feature='amount_spent',

               data=train_pay,

               label_df=train_label[['acc_id','survival_week']],

               title='Amount spent change over day by survival week',

               y_label='Amount spent')
train_combat_label = pd.merge(train_combat, train_label, on='acc_id', how='left')

day_week = train_combat_label[['acc_id','day','survival_week']].drop_duplicates()



plt.figure(figsize=(15,10))

ax1 = plt.subplot2grid((2,4), (0,0), colspan=2)

ax2 = plt.subplot2grid((2,4), (0,2), colspan=2)

ax3 = plt.subplot2grid((2,4), (1,0), colspan=4)

for i in range(10):

    day_df = day_week[day_week.survival_week==i]

    day_df = (day_df.day.value_counts() / day_df.shape[0]).reset_index()

    sns.lineplot(x='index', y='day', data=day_df, label='{} week'.format(i), ax=ax1)

ax1.set_title('Activity day distribution by Churn week')

ax1.set_xlabel('Day')

ax1.set_ylabel('Count')



for i in range(1,10):

    day_df = day_week[day_week.survival_week==i]

    day_df = (day_df.day.value_counts() / day_df.shape[0]).reset_index()

    sns.lineplot(x='index', y='day', data=day_df, label='{} week'.format(i), ax=ax2)

ax2.set_title('Activity day distribution by Churn week without 0 survival_week')

ax2.set_xlabel('Day')

ax2.set_ylabel('Count')





sns.lineplot(x='day', y='survival_week', data=day_week, ax=ax3)

ax3.set_title('Churn week distribution by activity day')

ax3.set_xlabel('Day')

ax3.set_ylabel('Survival week')

plt.show()
feature_by_day(feature='level',

               data=train_combat,

               label_df=train_label,

               title='Level change over day by survival week',

               y_label='Level',

               method='mean')
feature_by_day(feature='pledge_cnt',

               data=train_combat,

               label_df=train_label,

               title='Pledge cnt change over day by survival week',

               y_label='Pledge cnt')
feature_by_day(feature='random_attacker_cnt',

               data=train_combat,

               label_df=train_label,

               title='Random attacker cnt change over day by survival week',

               y_label='Random attacker cnt')
feature_by_day(feature='random_defender_cnt',

               data=train_combat,

               label_df=train_label,

               title='Random defender cnt change over day by survival week',

               y_label='Random defender cnt')
feature_by_day(feature='temp_cnt',

               data=train_combat,

               label_df=train_label,

               title='Temp cnt change over day by survival week',

               y_label='Temp cnt')
feature_by_day(feature='same_pledge_cnt',

               data=train_combat,

               label_df=train_label,

               title='Same pledge cnt change over day by survival week',

               y_label='Same pledge cnt')
feature_by_day(feature='etc_cnt',

               data=train_combat,

               label_df=train_label,

               title='etc cnt change over day by survival week',

               y_label='etc cnt')
feature_by_day(feature='num_opponent',

               data=train_combat,

               label_df=train_label,

               title='Num opponent change over day by survival week',

               y_label='Num opponent')