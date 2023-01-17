import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('bmh')

import matplotlib as mpl

import os

print(os.listdir("../input/bigcontest2019/"))

import gc



import warnings 

warnings.filterwarnings('ignore')
## Table Description Function

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





def multi_survival_kdeplot(df,col, subplot_x = 4, subplot_y = 3):

    for i,name in enumerate(col):

        plt.subplot(subplot_x,subplot_y,i+1)

        

        sns.kdeplot(df.loc[df['isSurvival'] == 1][name])

        sns.kdeplot(df.loc[df['isSurvival'] == 0][name])

        plt.title(f'Distribution of {name} by Survival')

        plt.legend(title = 'Surivival', loc = 'best', labels = ['Yes', 'No'])



        

def day_survival_boxplot(df, var, subplot_x = 4, subplot_y = 3):

    

    for i,j in enumerate(var):

        plt.subplot(subplot_x,subplot_y,i+1)

        

        sns.boxenplot(x = 'day', y = '{}'.format(j) ,hue = 'isSurvival', data = df )

        plt.title('Boxplot for {} by Day'.format(j))

        plt.xlabel('Day')

        

        

def day_survival_pointplot(df,col, subplot_x = 4, subplot_y = 3):

    

    for i,name in enumerate(col):

        plt.subplot(subplot_x,subplot_y,i+1)

        

        sns.pointplot(x = 'day', y = '{}'.format(name), hue = 'isSurvival', data = df)

        #sns.pointplot(x = 'week', y = '{}'.format(name), hue = 'isSurvival', data = df)

        plt.title('Pointplot of Mean for {} by Day'.format(name))

        #plt.title('Pointplot of Mean for {} by Week'.format(name))

        plt.xlabel('Week')

        

def week_survival_pointplot(df,col, subplot_x = 4, subplot_y = 3):

    

    for i,name in enumerate(col):

        plt.subplot(subplot_x,subplot_y,i+1)

        

        #sns.pointplot(x = 'day', y = '{}'.format(name), hue = 'isSurvival', data = df)

        sns.pointplot(x = 'week', y = '{}'.format(name), hue = 'isSurvival', data = df)

        #plt.title('Pointplot of Mean for {} by Day'.format(name))

        plt.title('Pointplot of Mean for {} by Week'.format(name))

        plt.xlabel('Week')
train_label = pd.read_csv('../input/bigcontest2019/train_label.csv')

train_act = pd.read_csv('../input/bigcontest2019/train_activity.csv')

train_trade = pd.read_csv('../input/bigcontest2019/train_trade.csv')

train_combat = pd.read_csv('../input/bigcontest2019//train_combat.csv')

train_pledge = pd.read_csv('../input/bigcontest2019/train_pledge.csv')

train_pay = pd.read_csv('../input/bigcontest2019/train_payment.csv')
train_label['isSurvival'] = train_label['survival_time'].map(lambda x:1 if x == 64 else 0)
print('Lable Quantiles by Surival Account : 1')

print(train_label.loc[train_label['isSurvival'] ==1][['amount_spent']].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))

print('Lable Quantiles by Surival Account : 0')

print(train_label.loc[train_label['isSurvival'] ==0][['amount_spent']].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
plt.figure(figsize = (12,6))



plt.subplot(121)

sns.kdeplot(train_label.loc[train_label['isSurvival'] ==1].amount_spent)

plt.title('Amount Spent by Surival Account')



plt.subplot(122)

sns.kdeplot(train_label.loc[train_label['isSurvival'] ==0].amount_spent,color = 'red')

plt.title('Amount Spent by Churning Account')
df_label_act = pd.merge(train_act,train_label, on = 'acc_id', how = 'left')



del train_act
resumetable(df_label_act)
drop_colname = {'day','acc_id','char_id','server','survival_time','amount_spent','isSurvival','rich_monster'}

use_colname = list(df_label_act.columns)

use_colname = [e for e in df_label_act if e not in drop_colname]



use_colname
print('Activity Quantiles by Surival Account : 1')

print(df_label_act.loc[df_label_act['isSurvival'] ==1][use_colname].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))

print('\n')

print('Activity Quantiles by Surival Account : 0')

print(df_label_act.loc[df_label_act['isSurvival'] ==0][use_colname].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
plt.figure(figsize = (24,15))



multi_survival_kdeplot(df_label_act,use_colname)
plt.figure(figsize = (24,20))



day_survival_pointplot(df_label_act,use_colname)
df_label_act['week'] = np.nan

df_label_act['week'][df_label_act['day'] <= 7] = 1

df_label_act['week'][(df_label_act['day'] >7) & (df_label_act['day'] <= 14)] = 2

df_label_act['week'][(df_label_act['day'] > 14) & (df_label_act['day'] <= 21)] = 3

df_label_act['week'][df_label_act['day'] > 21] = 4
plt.figure(figsize = (24,20))



week_survival_pointplot(df_label_act,use_colname)
total = len(df_label_act)



ax = sns.countplot(x = 'rich_monster', hue = 'isSurvival', data = df_label_act,palette = ["C1","C0"])

ax.legend(title = 'Survival', loc = 'best', labels = ['No','Yes'])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}%'.format(height/total*100),

           ha = 'center')
df_label_combat = pd.merge(train_combat,train_label, on = 'acc_id', how = 'left')
resumetable(df_label_combat)
drop_colname = {'day','acc_id','char_id','server','survival_time','amount_spent','isSurvival','class','level'}

use_colname = list(df_label_combat.columns)

use_colname = [e for e in df_label_combat if e not in drop_colname]



use_colname
print('Activity Quantiles by Surival Account : 1')

print(df_label_combat.loc[df_label_combat['isSurvival'] ==1][use_colname].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))

print('\n')

print('Activity Quantiles by Surival Account : 0')

print(df_label_combat.loc[df_label_combat['isSurvival'] ==0][use_colname].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
plt.figure(figsize = (24,15))



multi_survival_kdeplot(df_label_combat,use_colname)
plt.figure(figsize = (24,20))



day_survival_pointplot(df_label_combat,use_colname)
df_label_combat['week'] = np.nan

df_label_combat['week'][df_label_combat['day'] <= 7] = 1

df_label_combat['week'][(df_label_combat['day'] >7) & (df_label_combat['day'] <= 14)] = 2

df_label_combat['week'][(df_label_combat['day'] >14) & (df_label_combat['day'] <= 21)] = 3

df_label_combat['week'][df_label_combat['day'] > 21] = 4
plt.figure(figsize = (24,20))

week_survival_pointplot(df_label_combat,use_colname)
df_label_pledge = pd.merge(train_pledge, train_label, on ='acc_id', how = 'left')
drop_colname = {'day','acc_id','char_id','server','survival_time','amount_spent','isSurvival','pledge_id'}

use_colname = list(df_label_pledge.columns)

use_colname = [e for e in df_label_pledge if e not in drop_colname]



use_colname
print('Pledge Quantiles by Surival Account : 1')

print(df_label_pledge.loc[df_label_pledge['isSurvival'] ==1][use_colname].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))

print('\n')

print('Pledge Quantiles by Surival Account : 0')

print(df_label_pledge.loc[df_label_pledge['isSurvival'] ==0][use_colname].quantile([0.01,0.025,0.1,0.25,0.5,0.75,0.975,0.99,1]))
plt.figure(figsize = (24,15))



multi_survival_kdeplot(df_label_pledge,use_colname)
plt.figure(figsize = (24,20))



day_survival_pointplot(df_label_pledge,use_colname)
df_label_pledge['week'] = np.nan

df_label_pledge['week'][df_label_pledge['day'] <= 7] = 1

df_label_pledge['week'][(df_label_pledge['day'] >7) & (df_label_pledge['day'] <= 14)] = 2

df_label_pledge['week'][(df_label_pledge['day'] >14) & (df_label_pledge['day'] <= 21)] = 3

df_label_pledge['week'][df_label_pledge['day'] >21] = 4
plt.figure(figsize = (24,20))

week_survival_pointplot(df_label_pledge,use_colname)
df_combat_pledge = pd.merge(train_combat, train_pledge, on = ('acc_id','day','server','char_id'),how = 'left')
df_label_combat.sort_values(by = ['level'],ascending=False,inplace = True)

unique_df_label_combat = df_label_combat.drop_duplicates(subset = ['acc_id'],keep = 'first')
total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==0)&(unique_df_label_combat['day']==28)])

plt.figure(figsize = (13.5,4))



ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==0) & (unique_df_label_combat['day'] == 28)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot(New User) \n # 0: Churn | 1: Retention")



plt.figure(figsize = (12,10))

total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==0) & (unique_df_label_combat['week'] == 1)])

plt.subplot(221)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==0) & (unique_df_label_combat['week'] == 1)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 1 Week(New User) \n # 0: Churn | 1: Retention")



total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==0) & (unique_df_label_combat['week'] == 2)])

plt.subplot(222)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==0) & (unique_df_label_combat['week'] == 2)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 2 Week(New User) \n # 0: Churn | 1: Retention")



total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==0) & (unique_df_label_combat['week'] == 3)])

plt.subplot(223)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==0) & (unique_df_label_combat['week'] == 3)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 3 Week(New User) \n # 0: Churn | 1: Retention")





total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==0) & (unique_df_label_combat['week'] == 4)])

plt.subplot(224)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==0) & (unique_df_label_combat['week'] == 4)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 4 Week(New User) \n # 0: Churn | 1: Retention")

plt.tight_layout()
total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==17)&(unique_df_label_combat['day']==28)])

plt.figure(figsize = (13.5,2))



ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==17) & (unique_df_label_combat['day'] == 28)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot (High level) \n # 0: Churn | 1: Retention")



plt.figure(figsize = (12,10))

total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==17) & (unique_df_label_combat['week'] == 1)])

plt.subplot(221)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==17) & (unique_df_label_combat['week'] == 1)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 1 Week (High level) \n # 0: Churn | 1: Retention")



total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==17) & (unique_df_label_combat['week'] == 2)])

plt.subplot(222)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==17) & (unique_df_label_combat['week'] == 2)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 2 Week (High level) \n # 0: Churn | 1: Retention")



total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==17) & (unique_df_label_combat['week'] == 3)])

plt.subplot(223)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==17) & (unique_df_label_combat['week'] == 3)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 3 Week (High level) \n # 0: Churn | 1: Retention")





total = len(unique_df_label_combat.loc[(unique_df_label_combat['level']==17) & (unique_df_label_combat['week'] == 4)])

plt.subplot(224)

ax = sns.countplot(x = 'isSurvival',data = unique_df_label_combat.loc[(unique_df_label_combat['level'] ==17) & (unique_df_label_combat['week'] == 4)])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{:1.2f}'.format(height/total*100),

           ha = 'center')

ax.set_title("Retention or Churn Count plot by 4 Week\n # 0: Churn | 1: Retention")

plt.tight_layout()