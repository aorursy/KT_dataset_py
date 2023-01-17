import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib

matplotlib.rcParams['axes.unicode_minus']=False

plt.style.use('ggplot')

from sklearn.preprocessing import scale,minmax_scale

import os

import lightgbm as lgb

import xgboost as xgb
train_pay = pd.read_csv('../input/bigcontest2019/train_payment.csv')

print('train_pay.shape :',train_pay.shape)

train_pay.head()
train_pay.isna().sum()
train_lab = pd.read_csv('../input/bigcontest2019/train_label.csv')

print('train_lab.shape :',train_lab.shape)

train_lab.head()
train_lab.isna().sum()
train_lab['is_survival'] = train_lab['survival_time'].map(lambda x: 1 if x==64 else 0)

train_lab.head()
dp = train_pay.groupby('day')['amount_spent'].sum()

dp2 = train_pay.groupby('day')['amount_spent'].mean()
fig, ax = plt.subplots(2, 1, figsize=(15,12))

dp.plot(ax=ax[0])

ax[0].set(title='spent_sum by day', ylabel='spent_sum')

dp2.plot(ax=ax[1])

ax[1].set(title='spent_mean by day', ylabel='spent_mean')
tp = train_pay.groupby('acc_id')['day'].nunique().reset_index(name='day_count')

print(tp.shape)

tp.head()
print('28일간 가장 많이 결제한 사람의 결제 일 수 :', tp['day_count'].max())
print('20일 결제한 사람 수 :', tp[tp['day_count']==20].shape[0])
print('28일 동안 하루만 결제한 사람 수 :', tp[tp['day_count']==1].shape[0])
plt.figure(figsize=(15,7))

s = sns.countplot(tp['day_count'])

s.set(ylabel='count')

s.set(title='day_count dist.')

for p in s.patches:

    height = p.get_height()

    s.text(p.get_x() + p.get_width()/2.,

           height + 3,

           '{}'.format(int(height)),

           ha = 'center')
lab = pd.merge(train_lab, tp, on='acc_id', how='left')

lab.head()
lab.isna().sum()
lab = lab.fillna(0)

lab.isna().sum()
fig, ax = plt.subplots(1, 2, figsize=(15,7))

sns.scatterplot(x='day_count', y='amount_spent', data=lab, ax=ax[0])

ax[0].set(title='amount_spent by day_count')

sns.scatterplot(x='day_count', y='survival_time', data=lab, ax=ax[1])

ax[1].set(title='survival_time by day_count')
sns.heatmap(data=lab.drop('acc_id',axis=1).corr(), annot=True, fmt='.2f')
pay = train_pay.groupby('acc_id')['amount_spent'].mean().reset_index(name='daily_spent')

print('pay.shape :', pay.shape)

pay.head()
plt.figure(figsize=(15,7))

ss = sns.distplot(pay['daily_spent'])

ss.set(title='daily_spent dist.')

ss.set(ylabel='density')
label = pd.merge(lab, pay, on='acc_id', how='left')

label.head()
label.isna().sum()
label = label.fillna(0)

label.isna().sum()
print('관측기간(70일)동안 결제한 적 없는 사람 수 :', label[label['amount_spent']==0].shape[0])

print('활동데이터 수집기간(28일)동안 결제한 적 없는 사람 수 :', label[label['daily_spent']==0].shape[0])

print('활동 & 관측 둘 다 결제한 적 없는 사람 수 :', label[(label['daily_spent']==0) & (label['amount_spent']==0)].shape[0])
sns.heatmap(data=label.drop('acc_id',axis=1).corr(), annot=True, fmt='.2f')
fig, ax = plt.subplots(1, 2, figsize=(15,7))

sns.regplot(x='amount_spent', y='daily_spent', data=label, ax=ax[0])

ax[0].set(title='regplot by label_spent and payment_spent')

sns.regplot(x='day_count', y='daily_spent', data=label, ax=ax[1])

ax[1].set(title='regplot by day_count and payment_spent')
fig, ax = plt.subplots(1, 2, figsize=(15,7))

sns.kdeplot(label[label['is_survival']==1]['amount_spent'], label='survive', ax=ax[0])

sns.kdeplot(label[label['is_survival']==0]['amount_spent'], label='leave', ax=ax[0])

ax[0].set(xlabel='amount_spent',

          ylabel='density',

          title='amount_spent by is_survival')

sns.kdeplot(label[label['is_survival']==1]['daily_spent'], label='survive', ax=ax[1])

sns.kdeplot(label[label['is_survival']==0]['daily_spent'], label='leave', ax=ax[1])

ax[1].set(xlabel='daily_spent',

          ylabel='density',

          title='daily_spent by is_survival')
plt.figure(figsize=(15,7))

s = sns.scatterplot(x='survival_time', y='daily_spent', data=label)

s.set(title='daily_spent by survival_time')
fig, ax = plt.subplots(2, 2, figsize=(15,12))

sns.countplot(label[label['amount_spent']==0]['is_survival'], ax=ax[0,0])

ax[0,0].set(title='is_survival by (amount_spent = 0)')

sns.countplot(label[label['daily_spent']==0]['is_survival'], ax=ax[0,1])

ax[0,1].set(title='is_survival by (daily_spent = 0)')

sns.distplot(label[label['amount_spent']==0]['survival_time'], ax=ax[1,0])

ax[1,0].set(ylabel='density', title='survival_time by (amount_spent = 0)')

sns.distplot(label[label['daily_spent']==0]['survival_time'], ax=ax[1,1])

ax[1,1].set(ylabel='density', title='survival_time by (daily_spent = 0)')
pay2 = train_pay.groupby('acc_id')['amount_spent'].sum().reset_index(name='sum_spent')

pay2.head()
label = pd.merge(label, pay2, on='acc_id', how='left')

label.head()
label.isna().sum()
label = label.fillna(0)

label.isna().sum()
plt.figure(figsize=(15,7))

s = sns.distplot(label['sum_spent'])

s.set(title='sum_spent dist.', ylabel='density')
sns.heatmap(data=label.drop('acc_id',axis=1).corr(), annot=True, fmt='.2f')
fig, ax = plt.subplots(1, 2, figsize=(15,7))

sns.kdeplot(label[label['is_survival']==1]['daily_spent'], label='survive', ax=ax[0])

sns.kdeplot(label[label['is_survival']==0]['daily_spent'], label='leave', ax=ax[0])

ax[0].set(xlabel='daily_spent',

          ylabel='density',

          title='daily_spent by is_survival')

sns.kdeplot(label[label['is_survival']==1]['sum_spent'], label='survive', ax=ax[1])

sns.kdeplot(label[label['is_survival']==0]['sum_spent'], label='leave', ax=ax[1])

ax[1].set(xlabel='sum_spent',

          ylabel='density',

          title='sum_spent by is_survival')
plt.figure(figsize=(15,7))

s = sns.scatterplot(x='survival_time', y='sum_spent', data=label)

s.set(title='sum_spent by survival_time')
plt.figure(figsize=(15,7))

s = sns.regplot(x='amount_spent', y='sum_spent', data=label)

s.set(title='regplot by amount_spent and sum_spent')
label.head()
label['sum_amount'] = label['survival_time'] * label['amount_spent']

label.head()
plt.figure(figsize=(15,7))

sns.distplot(label['sum_amount'])
plt.figure(figsize=(15,7))

sns.scatterplot(x='survival_time', y='sum_amount', data=label)
sns.heatmap(data=label.drop('acc_id',axis=1).corr(), annot=True, fmt='.2f')
plt.figure(figsize=(15,7))

s = sns.regplot(x='sum_amount', y='sum_spent', data=label)

s.set(title='regplot by sum_amount and sum_spent')