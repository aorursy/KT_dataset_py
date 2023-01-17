import numpy as np

import os

import pandas as pd
DATA_PATH='/kaggle/input/abstract-data-set-for-credit-card-fraud-detection/creditcardcsvpresent.csv'
import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt
entire_df = pd.read_csv(DATA_PATH)
def regularization_l2(data_arr):

    return (data_arr/ np.linalg.norm(data_arr))
def regularization_mean_std_d(data_arr):

    mean = np.mean(data_arr)

    std = np.std(data_arr) 

    regularaz_arr = (data_arr - mean)/std

    return regularaz_arr
def sub_mean(data_arr):

    mean = np.mean(data_arr)

    regularaz_arr = (data_arr - mean)

    return regularaz_arr
def max_min_reg(data_arr):

    max_ = np.max(data_arr)

    min_ = np.min(data_arr) 

    regularaz_arr = (data_arr - min_)/(max_-min_)

    return regularaz_arr
entire_df.head()
entire_df.describe()
entire_df.isnull().sum()
f, ax = plt.subplots(1, 2, figsize=(18, 8))







entire_df['isFradulent'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - isFradulent')

ax[0].set_ylabel('')

sns.countplot('isFradulent', data=entire_df, ax=ax[1])

ax[1].set_title('Count plot - isFradulent')



plt.show()
dropcolums= ['Merchant_id','Transaction date']
entire_df = entire_df.drop(dropcolums,axis=1)
from sklearn.model_selection import train_test_split

df_train,df_test = train_test_split(entire_df,test_size=0.25, random_state=2020)
df_train['isFradulent']= df_train['isFradulent'].replace('N',0)

df_train['isFradulent']=df_train['isFradulent'].replace('Y',1)
df_test['isFradulent']= df_test['isFradulent'].replace('N',0)

df_test['isFradulent']=df_test['isFradulent'].replace('Y',1)
df_train['Average Amount/transaction/day'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day'],ax=ax)

sns.kdeplot(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day'],ax=ax)

plt.legend(['isFradulent==0','isFradulent==1'])

plt.title('Ogriginal Average Amount/transaction/day distributution')

plt.show()
plt.hist(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day'])

plt.hist(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
per_df_tr_of_75 = df_train['Average Amount/transaction/day'].quantile(q=0.75, interpolation='nearest')

per_df_tr_of_25 = df_train['Average Amount/transaction/day'].quantile(q=0.25, interpolation='nearest')

out_line = (per_df_tr_of_75-per_df_tr_of_25)*3

df_train = df_train.drop(df_train.loc[df_train['Average Amount/transaction/day']>out_line].index.values.astype(int))
plt.hist(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day'])

plt.hist(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day']),ax = ax[0][0])

sns.kdeplot(np.log(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day']),ax=ax[0][0])

ax[0][0].legend(['isFradulent==0','isFradulent==1'])

ax[0][0].set_title('Log')



sns.kdeplot(np.sin(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day']),ax = ax[0][1])

sns.kdeplot(np.sin(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day']),ax=ax[0][1])

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin ')



sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day']),ax = ax[0][2])

sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day']),ax=ax[0][2])

ax[0][2].legend(['target==0','target==1'])

ax[0][2].set_title('sub_mean ')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day']),ax = ax[1][0])

sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day']),ax=ax[1][0])

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg ')



sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day']),ax = ax[1][1])

sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day']),ax=ax[1][1])

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg ')



sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==0]['Average Amount/transaction/day']),ax = ax[1][2])

sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==1]['Average Amount/transaction/day']),ax=ax[1][2])

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max reg')

plt.show()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['isFradulent']==0]['Transaction_amount'],ax = ax)

sns.kdeplot(df_train[df_train['isFradulent']==1]['Transaction_amount'],ax=ax)

plt.legend(['isFradulent==0','isFradulent==1'])

plt.title('Transaction_amount distributution')

plt.show()
plt.hist(df_train[df_train['isFradulent']==0]['Transaction_amount'])

plt.hist(df_train[df_train['isFradulent']==1]['Transaction_amount'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['isFradulent']==0]['Transaction_amount']),ax = ax[0][0])

sns.kdeplot(np.log(df_train[df_train['isFradulent']==1]['Transaction_amount']),ax=ax[0][0])

ax[0][0].legend(['isFradulent==0','isFradulent==1'])

ax[0][0].set_title('Log')



sns.kdeplot(np.sin(df_train[df_train['isFradulent']==0]['Transaction_amount']),ax = ax[0][1])

sns.kdeplot(np.sin(df_train[df_train['isFradulent']==1]['Transaction_amount']),ax=ax[0][1])

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin ')



sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==0]['Transaction_amount']),ax = ax[0][2])

sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==1]['Transaction_amount']),ax=ax[0][2])

ax[0][2].legend(['target==0','target==1'])

ax[0][2].set_title('sub_mean ')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==0]['Transaction_amount']),ax = ax[1][0])

sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==1]['Transaction_amount']),ax=ax[1][0])

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg ')



sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==0]['Transaction_amount']),ax = ax[1][1])

sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==1]['Transaction_amount']),ax=ax[1][1])

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg ')



sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==0]['Transaction_amount']),ax = ax[1][2])

sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==1]['Transaction_amount']),ax=ax[1][2])

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max reg')

plt.show()
df_train['Is declined'].describe()
df_train['Total Number of declines/day'].describe()
fig,ax = plt.subplots(1,2,figsize=(9,5))

sns.countplot(df_train[df_train['isFradulent']==0]['Total Number of declines/day'],ax =ax[0])

sns.countplot(df_train[df_train['isFradulent']==1]['Total Number of declines/day'],ax=ax[0])

ax[0].legend(['isFradulent==0','isFradulent==1'])





ax[1].hist(df_train[df_train['isFradulent']==0]['Total Number of declines/day'])

ax[1].hist(df_train[df_train['isFradulent']==1]['Total Number of declines/day'])

ax[1].legend(['isFradulent==0','isFradulent==1'])

plt.show()
sns.violinplot('Total Number of declines/day','Is declined',hue='isFradulent',data=df_train,scale='count')

fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['isFradulent']==0]['Total Number of declines/day'],ax = ax,cumulative=True, bw=1.5)

sns.kdeplot(df_train[df_train['isFradulent']==1]['Total Number of declines/day'],ax=ax,cumulative=True, bw=1.5)

plt.legend(['isFradulent==0','isFradulent==1'])

plt.title('Transaction_amount distributution')

plt.show()
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['isFradulent']==0]['Total Number of declines/day']),ax = ax[0][0],cumulative=True, bw=1.5)

sns.kdeplot(np.log(df_train[df_train['isFradulent']==1]['Total Number of declines/day']),ax=ax[0][0],cumulative=True, bw=1.5)

ax[0][0].legend(['isFradulent==0','isFradulent==1'])

ax[0][0].set_title('Log')



sns.kdeplot(np.sin(df_train[df_train['isFradulent']==0]['Total Number of declines/day']),ax = ax[0][1],cumulative=True, bw=1.5)

sns.kdeplot(np.sin(df_train[df_train['isFradulent']==1]['Total Number of declines/day']),ax=ax[0][1],cumulative=True, bw=1.5)

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin ')



sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==0]['Total Number of declines/day']),ax = ax[0][2],cumulative=True, bw=1.5)

sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==1]['Total Number of declines/day']),ax=ax[0][2],cumulative=True, bw=1.5)

ax[0][2].legend(['target==0','target==1'])

ax[0][2].set_title('sub_mean ')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==0]['Total Number of declines/day']),ax = ax[1][0],cumulative=True, bw=1.5)

sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==1]['Total Number of declines/day']),ax=ax[1][0],cumulative=True, bw=1.5)

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg ')



sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==0]['Total Number of declines/day']),ax = ax[1][1],cumulative=True, bw=1.5)

sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==1]['Total Number of declines/day']),ax=ax[1][1],cumulative=True, bw=1.5)

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg ')



sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==0]['Total Number of declines/day']),ax = ax[1][2],cumulative=True, bw=1.5)

sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==1]['Total Number of declines/day']),ax=ax[1][2],cumulative=True, bw=1.5)

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max reg')

plt.show()
df_train['Total Number of declines/day'] = sub_mean(df_train['Total Number of declines/day'])
df_test['Total Number of declines/day'] = sub_mean(df_test['Total Number of declines/day'])
df_train['Is declined']= df_train['Is declined'].replace('N',-1)

df_train['Is declined']=df_train['Is declined'].replace('Y',1)
df_test['Is declined']= df_test['Is declined'].replace('N',-1)

df_test['Is declined']=df_test['Is declined'].replace('Y',1)
df_train['Is declined'] = regularization_mean_std_d(df_train['Total Number of declines/day']) +df_train['Is declined']
df_test['Is declined'] = regularization_mean_std_d(df_test['Total Number of declines/day']) +df_test['Is declined']
plt.hist(df_train[df_train['isFradulent']==0]['Is declined'])

plt.hist(df_train[df_train['isFradulent']==1]['Is declined'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
df_train[df_train['isFradulent']==0]['Is declined'].describe()
df_train[df_train['isFradulent']==1]['Is declined'].describe()
df_train['isForeignTransaction'].describe()
plt.hist(df_train[df_train['isFradulent']==0]['isForeignTransaction'])

plt.hist(df_train[df_train['isFradulent']==1]['isForeignTransaction'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
pd.crosstab(df_train['isForeignTransaction'], df_train['isFradulent'], margins=True).style.background_gradient(cmap='summer_r')
df_train['isHighRiskCountry'].describe()
plt.hist(df_train[df_train['isFradulent']==0]['isHighRiskCountry'])

plt.hist(df_train[df_train['isFradulent']==1]['isHighRiskCountry'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
pd.crosstab(df_train['isHighRiskCountry'], df_train['isFradulent'], margins=True).style.background_gradient(cmap='summer_r')
fig,ax = plt.subplots(1,2,figsize=(16,7))

sns.violinplot(x='isForeignTransaction',y='isFradulent',data=df_train,split=True,ax=ax[0])

ax[0].set_yticks(range(0,3,1))

ax[0].set_title('isForeignTransaction with isFradulent')

sns.violinplot(x='isHighRiskCountry',y='isFradulent',data=df_train,split=True,ax=ax[1])

ax[1].set_yticks(range(0,3,1))

ax[1].set_title('isHighRiskCountry with isFradulent')

plt.show()
df_train['isHighRiskCountry']= df_train['isHighRiskCountry'].replace('N',-1)

df_train['isHighRiskCountry']=df_train['isHighRiskCountry'].replace('Y',1)



df_train['isForeignTransaction']= df_train['isForeignTransaction'].replace('N',1)

df_train['isForeignTransaction']=df_train['isForeignTransaction'].replace('Y',0)

df_test['isHighRiskCountry']= df_test['isHighRiskCountry'].replace('N',-1)

df_test['isHighRiskCountry']=df_test['isHighRiskCountry'].replace('Y',1)



df_test['isForeignTransaction']= df_test['isForeignTransaction'].replace('N',1)

df_test['isForeignTransaction']=df_test['isForeignTransaction'].replace('Y',0)

df_train['Daily_chargeback_avg_amt'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt'],ax=ax,cumulative=True, bw=1.5)

sns.kdeplot(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt'],ax=ax,cumulative=True, bw=1.5)

plt.legend(['isFradulent==0','isFradulent==1'])

plt.title('Daily_chargeback_avg_amt distributution')

plt.show()
plt.hist(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt'])

plt.hist(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
pd.crosstab(df_train['Daily_chargeback_avg_amt'], df_train['isFradulent'], margins=True).style.background_gradient(cmap='summer_r')
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt']),ax = ax[0][0],cumulative=True, bw=1.5)

sns.kdeplot(np.log(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt']),ax=ax[0][0],cumulative=True, bw=1.5)

ax[0][0].legend(['isFradulent==0','isFradulent==1'])

ax[0][0].set_title('Log')



sns.kdeplot(np.sin(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt']),ax = ax[0][1],cumulative=True, bw=1.5)

sns.kdeplot(np.sin(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt']),ax=ax[0][1],cumulative=True, bw=1.5)

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin ')



sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt']),ax = ax[0][2],cumulative=True, bw=1.5)

sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt']),ax=ax[0][2],cumulative=True, bw=1.5)

ax[0][2].legend(['target==0','target==1'])

ax[0][2].set_title('sub_mean ')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt']),ax = ax[1][0],cumulative=True, bw=1.5)

sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt']),ax=ax[1][0],cumulative=True, bw=1.5)

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg ')



sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt']),ax = ax[1][1],cumulative=True, bw=1.5)

sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt']),ax=ax[1][1],cumulative=True, bw=1.5)

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg ')



sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==0]['Daily_chargeback_avg_amt']),ax = ax[1][2],cumulative=True, bw=1.5)

sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==1]['Daily_chargeback_avg_amt']),ax=ax[1][2],cumulative=True, bw=1.5)

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max reg')

plt.show()
df_train['6_month_avg_chbk_amt'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt'],ax=ax,cumulative=True, bw=1.5)

sns.kdeplot(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt'],ax=ax,cumulative=True, bw=1.5)

plt.legend(['isFradulent==0','isFradulent==1'])

plt.title('6_month_avg_chbk_amt distributution')

plt.show()
plt.hist(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt'])

plt.hist(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt']),ax = ax[0][0],cumulative=True, bw=1.5)

sns.kdeplot(np.log(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt']),ax=ax[0][0],cumulative=True, bw=1.5)

ax[0][0].legend(['isFradulent==0','isFradulent==1'])

ax[0][0].set_title('Log')



sns.kdeplot(np.sin(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt']),ax = ax[0][1],cumulative=True, bw=1.5)

sns.kdeplot(np.sin(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt']),ax=ax[0][1],cumulative=True, bw=1.5)

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin ')



sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt']),ax = ax[0][2],cumulative=True, bw=1.5)

sns.kdeplot(sub_mean(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt']),ax=ax[0][2],cumulative=True, bw=1.5)

ax[0][2].legend(['target==0','target==1'])

ax[0][2].set_title('sub_mean ')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt']),ax = ax[1][0],cumulative=True, bw=1.5)

sns.kdeplot(regularization_mean_std_d(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt']),ax=ax[1][0],cumulative=True, bw=1.5)

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg ')



sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt']),ax = ax[1][1],cumulative=True, bw=1.5)

sns.kdeplot(regularization_l2(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt']),ax=ax[1][1],cumulative=True, bw=1.5)

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg ')



sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt']),ax = ax[1][2],cumulative=True, bw=1.5)

sns.kdeplot(max_min_reg(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt']),ax=ax[1][2],cumulative=True, bw=1.5)

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max reg')

plt.show()
df_train['6-month_chbk_freq'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['isFradulent']==0]['6-month_chbk_freq'],ax=ax,cumulative=True, bw=1.5)

sns.kdeplot(df_train[df_train['isFradulent']==1]['6-month_chbk_freq'],ax=ax,cumulative=True, bw=1.5)

plt.legend(['isFradulent==0','isFradulent==1'])

plt.title('6-month_chbk_freq distributution')

plt.show()
plt.hist(df_train[df_train['isFradulent']==0]['6-month_chbk_freq'])

plt.hist(df_train[df_train['isFradulent']==1]['6-month_chbk_freq'])

plt.legend(['isFradulent==0','isFradulent==1'])

plt.show()
sns.barplot(x="6-month_chbk_freq", y="6_month_avg_chbk_amt", data=df_train)
df_train['6_month_avg_chbk_amt'].describe()
fig,ax = plt.subplots(1,2,figsize=(9,5))

sns.kdeplot(df_train[df_train['isFradulent']==0]['6_month_avg_chbk_amt'],cumulative=True, bw=1.5,ax=ax[0])

sns.kdeplot(df_train[df_train['isFradulent']==1]['6_month_avg_chbk_amt'],cumulative=True, bw=1.5,ax=ax[0])

ax[0].legend(['isFradulent==0','isFradulent==1'])



ax[1].hist(df_train[df_train['isFradulent']==0]['6-month_chbk_freq'])

ax[1].hist(df_train[df_train['isFradulent']==1]['6-month_chbk_freq'])

ax[1].legend(['isFradulent==0','isFradulent==1'])

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')



xs = df_train['6-month_chbk_freq']

ys = df_train['6_month_avg_chbk_amt']

zs = df_train['isFradulent']

ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')



ax.set_xlabel('6-month_chbk_freq')

ax.set_ylabel('6_month_avg_chbk_amt')

ax.set_zlabel('isFradulent')

ax.scatter(xs, ys, zs, c = zs, s= 50, alpha=0.5, cmap=plt.cm.Greens)

#ax.view_init(20, 45)

plt.show()

df_train['Average Amount/transaction/day'] = regularization_l2(df_train['Average Amount/transaction/day'])

df_train['Transaction_amount']= regularization_l2(df_train['Transaction_amount'])

df_train['Daily_chargeback_avg_amt']=sub_mean(df_train['Daily_chargeback_avg_amt'])

df_train['6_month_avg_chbk_amt']=regularization_mean_std_d(df_train['6_month_avg_chbk_amt'])
df_test['Average Amount/transaction/day'] = regularization_l2(df_test['Average Amount/transaction/day'])

df_test['Transaction_amount']= regularization_l2(df_test['Transaction_amount'])

df_test['Daily_chargeback_avg_amt']=sub_mean(df_test['Daily_chargeback_avg_amt'])

df_test['6_month_avg_chbk_amt']=regularization_mean_std_d(df_test['6_month_avg_chbk_amt'])
y_train = np.asarray(df_train['isFradulent'])

x_train = np.asarray(df_train.drop(['isFradulent'],axis=1))
y_test = np.asarray(df_test['isFradulent'])

x_test = np.asarray(df_test.drop(['isFradulent'],axis=1))
from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, f1_score, precision_score
model_performance_list = []

model_list = ['RandomForest','Xgboost']
sf_rf_train_res_ensem = np.zeros(len(x_train))

sf_rf_test_res_ensem = np.zeros(len(x_test))

skfold = StratifiedKFold(n_splits=5)

rf_c = RandomForestClassifier(    

            n_estimators= 100,

            max_depth=5,

            ccp_alpha=1e-07,

         random_state=2020)

for tr_index,test_index in skfold.split(x_train,y_train):

    rf_c.fit(x_train[tr_index],y_train[tr_index])

    y_pred =  rf_c.predict(x_train)

    sf_rf_train_res_ensem+=y_pred

    sf_rf_test_res_ensem+=rf_c.predict(x_test)

sf_rf_train_res_ensem =np.round(sf_rf_train_res_ensem/5)

sf_rf_test_res_ensem =np.round(sf_rf_test_res_ensem/5)

#rf_c.fit(x_train,y_train)
print("acc_train score : ", accuracy_score(sf_rf_train_res_ensem,y_train),",f1_score :",f1_score(sf_rf_train_res_ensem,y_train))

print("acc_train score : ", accuracy_score(sf_rf_test_res_ensem,y_test),",f1_score :",f1_score(sf_rf_test_res_ensem,y_test))

model_performance_list.append(f1_score(sf_rf_test_res_ensem,y_test))
xgb_c = XGBClassifier(    

    learning_rate =0.005,

    n_estimators=100,

    max_depth=10,

    min_child_weight=8,

    gamma=0.0008,

    reg_alpha=1e-05,

    subsample=0.99,

    colsample_bytree=0.8,

    objective= 'binary:logistic',

    nthread=-1,

    scale_pos_weight=1,

    seed=2020)
skfold = StratifiedKFold(n_splits=5)

sf_xgb_train_res_ensem = np.zeros(len(x_train))

sf_xgb_test_res_ensem =np.zeros(len(x_test))

for tr_index,test_index in skfold.split(x_train,y_train):

    xgb_c.fit(x_train[tr_index],y_train[tr_index])

    y_pred =  xgb_c.predict(x_train)

    sf_xgb_train_res_ensem+=y_pred

    sf_xgb_test_res_ensem+=xgb_c.predict(x_test)

sf_xgb_train_res_ensem =  np.round(sf_xgb_train_res_ensem/4)

sf_xgb_test_res_ensem = np.round(sf_xgb_test_res_ensem/4)
from sklearn.metrics import accuracy_score, f1_score, precision_score

print("acc_train score : ", accuracy_score(sf_xgb_train_res_ensem,y_train),",f1_score :",f1_score(sf_xgb_train_res_ensem,y_train))

print("acc_train score : ", accuracy_score(sf_xgb_test_res_ensem,y_test),",f1_score :",f1_score(sf_xgb_test_res_ensem,y_test))

model_performance_list.append(f1_score(sf_xgb_test_res_ensem,y_test))
from pandas import Series
plt.figure(figsize=(8, 8))

sns.barplot(model_list,model_performance_list)

plt.ylabel('performance')

plt.ylabel('pre-train model list')

plt.show()
print("rf+xgb ensemble f1_score :",f1_score(np.round((sf_xgb_test_res_ensem+sf_rf_test_res_ensem)/2),y_test))
feature_importance = xgb_c.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_train.drop(['isFradulent'],axis=1).columns)

plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('xgboost Feature')

plt.show()
Series_feat_imp = Series(feature_importance, index=df_train.drop(['isFradulent'],axis=1).columns)

plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Random forest Feature')

plt.show()
from sklearn.metrics import roc_curve

fpr1, tpr1, thresholds1 = roc_curve(y_test, xgb_c.predict(x_test))

fpr2, tpr2, thresholds1 = roc_curve(y_test, rf_c.predict(x_test))



plt.plot(fpr1, tpr1, 'o-', ms=2, label="Logistic Regression")

plt.plot(fpr2, tpr2, 'o-', ms=2, label="Kernel SVM")

plt.legend()

plt.plot([0, 1], [0, 1], 'k--', label="random guess")

plt.xlabel('위양성률(Fall-Out)')

plt.ylabel('재현률(Recall)')

plt.title('ROC 커브')

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test,sf_xgb_test_res_ensem))