import numpy as np

import pandas as pd



# graph

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



# split validation set

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
label = pd.read_csv('/kaggle/input/train_label.csv')

pledge = pd.read_csv('/kaggle/input/train_pledge.csv')

activity = pd.read_csv('/kaggle/input/train_activity.csv')

combat = pd.read_csv('/kaggle/input/train_combat.csv')

payment = pd.read_csv('/kaggle/input/train_payment.csv')

print('label.shape: ',label.shape)

print('pledge.shape: ',pledge.shape)

print('activity.shape: ',activity.shape)

print('combat.shape: ',combat.shape)

print('payment.shape: ',payment.shape)
train_set = [pledge, activity, combat]

total_user_char_df = pd.DataFrame()

for df in train_set:

    user_char_df = df[['acc_id','char_id']].drop_duplicates()

    total_user_char_df = pd.concat([total_user_char_df, user_char_df], axis=0)

total_user_char_df = total_user_char_df.drop_duplicates()

total_user_char_cnt_df = total_user_char_df.acc_id.value_counts().reset_index().rename(columns={'index':'acc_id','acc_id':'char_cnt'})

print('유저&캐릭터 DataFrame 크기: ',total_user_char_df.shape)

print('유저별 캐릭터 수 DataFrame 크기: ', total_user_char_cnt_df.shape)
sns.kdeplot(total_user_char_cnt_df.char_cnt, shade=True, label='Number of Character')

plt.title('Number of Character by User Account Distribution')

plt.show()
total_user_char_cnt_df.char_cnt.describe()
train_set = [pledge, activity, combat, payment]

total_user_day_df = pd.DataFrame()

for df in train_set:

    user_day_df = df[['acc_id','day']].drop_duplicates()

    total_user_day_df = pd.concat([total_user_day_df, user_day_df], axis=0)

total_user_day_df = total_user_day_df.drop_duplicates()

total_user_day_cnt_df = total_user_day_df.acc_id.value_counts().reset_index().rename(columns={'index':'acc_id','acc_id':'day_cnt'})

print('total_user_day_df.shape: ',total_user_day_df.shape)

print('total_user_day_cnt_df.shape: ', total_user_day_cnt_df.shape)
sns.kdeplot(total_user_day_cnt_df.day_cnt, shade=True, label='Activity Days')

plt.title('Activity Days by User Account')

plt.show()
total_user_day_cnt_df.day_cnt.describe()
label_set = label.copy()

label_set['survived'] = label_set.survival_time.apply(lambda x: 1 if x == 64 else 0)

label_set = pd.merge(label_set, total_user_char_cnt_df, on='acc_id', how='inner')

label_set = pd.merge(label_set, total_user_day_cnt_df, on='acc_id', how='inner')

print('label_set shape: ',label_set.shape)
X = label_set.drop(['survival_time','amount_spent'], axis=1)

y = label_set[['survival_time','amount_spent']]

x_train, x_valid, y_train, y_valid = train_test_split(X, 

                                                     y,

                                                     test_size=0.3,

                                                     random_state=2019,

                                                     stratify=y.survival_time)
print('Number of user account in train set: ',x_train.shape[0])

print('Number of user account in validation set: ',x_valid.shape[0])
print('좌: Train set / 우: Validation set')

pd.concat([y_train.describe(), y_valid.describe()], axis=1)
f, ax = plt.subplots(1,2, figsize=(15,5))

sns.kdeplot(y_train.survival_time, shade=True, ax=ax[0], label='Train')

sns.kdeplot(y_valid.survival_time, shade=True, ax=ax[0], label='Validation')

ax[0].set_title('Survival Time Distribution of Train/Validation Set')



sns.kdeplot(y_train.amount_spent, shade=True, ax=ax[1], label='Train')

sns.kdeplot(y_valid.amount_spent, shade=True, ax=ax[1], label='Validation')

ax[1].set_title('Amount Spent Distribution of Train.Validation Set')
train = pd.concat([x_train, y_train], axis=1)

valid = pd.concat([x_valid, y_valid], axis=1)

train['set'] = 'Train'

valid['set'] = 'Validation'

total = pd.concat([train,valid], axis=0)

print('total.shape: ',total.shape)
f, ax = plt.subplots(1,3, figsize=(25,5))

sns.countplot(x='survived', hue='set', data=total, ax=ax[0])

ax[0].legend(['Train','Validation'])

ax[0].set_xticklabels(['Survived','Not Survived'])

ax[0].set_xlabel('')

ax[0].set_title('Frequency of Survived in Train/Validation Set')



sns.kdeplot(data=total.loc[total.set=='Train','char_cnt'], ax=ax[1], shade=True, label='Train')

sns.kdeplot(data=total.loc[total.set=='Validation','char_cnt'], ax=ax[1], shade=True, label='Validation')

ax[1].set_title('Number of Character by User Account Distribution in Train/Validation Set')



sns.kdeplot(data=total.loc[total.set=='Train','day_cnt'], ax=ax[2], shade=True, label='Train')

sns.kdeplot(data=total.loc[total.set=='Validation','day_cnt'], ax=ax[2], shade=True, label='Validation')

ax[2].set_title('Activity Days by User Account in Train/Validation Set')

plt.show()
train_valid_user_id = total[['acc_id','set']]

print('train_valid_user_id shape: ',train_valid_user_id.shape)
train_valid_user_id.to_csv('train_valid_user_id.csv', index=False)