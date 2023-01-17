# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from pandas_profiling import ProfileReport



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
purchases = pd.read_csv(r'../input/purchuses/purchases_.csv',sep=',')

print('Размерность таблицы: '+str(purchases.shape))

purchases.head()
purchases_profile = ProfileReport(purchases, title='Purchases Report',minimal=True)
purchases_profile.to_notebook_iframe()
purchases['time'].plot.kde()
fig = plt.figure()

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



purchases[purchases['price']<400]['price'].plot.hist(bins=20,ax=ax1,title='Price 20 bins')

purchases[purchases['price']<400]['price'].plot.hist(bins=60,ax=ax2,title='Price 60 bins')
sns.relplot(x="time", y="price", hue="consumption_mode", data=purchases)
sns.relplot(x="time", y="price", hue="consumption_mode", data=purchases[purchases['price']<100])
sns.jointplot("time", "price", data=purchases[purchases['price']<100], kind="hex")
users = pd.read_csv(r'../input/purchuses/users_.csv',sep=',')

print('Размерность таблицы: '+str(users.shape))

users.head()
users_profile = ProfileReport(users, title='Users Report',minimal=True)
users_profile.to_notebook_iframe()
tmp_user = users[users['user_uid']=='b6c9ba2b342248e18633b95d9fc1333d'].sort_values(by=['ts'])

tmp_purchase = purchases[purchases['user_uid']=='26890a8487248806518c97c316ba1d05'].sort_values(by=['time'])

tmp_user.head()
A = set(purchases['user_uid'])

B = set(users['user_uid'])

A &= B
users2 = users[users['user_uid'].isin(A)]

print(users2.shape)

purchases2 = purchases[purchases['user_uid'].isin(A)]

print(purchases2.shape)
users2['user_uid'].value_counts().head()
users2[users2['user_uid']=='c64be9acc99af59b0e04d2ce016e057e']
purchases2[purchases2['user_uid']=='c64be9acc99af59b0e04d2ce016e057e'].sort_values(by=['time'])
error_users = users2[['tag','user_uid']].groupby('user_uid').nunique().reset_index().query('tag>1').sort_values('tag')

print(error_users.shape)

print(error_users.head())

error_users_id = set(error_users['user_uid'])
users2 = users2[~users2['user_uid'].isin(error_users_id)]

purchases2 = purchases2[~purchases2['user_uid'].isin(error_users_id)]
min_ts = users2[['user_uid','ts']].groupby('user_uid').min().reset_index().rename(columns={"ts": "ts_min"})

users3 = users2.merge(min_ts,how = 'left',on='user_uid')

users3 = users3[users3['ts']==users3['ts_min']]

print(users2.shape)

print(users3.shape)

print(users3.drop_duplicates().shape)
users3.drop('ts_min',axis=1,inplace=True)

del users,users2,purchases
purchases2 = purchases2.merge(users3,how='left',on='user_uid')

purchases2.head()
final_profile = ProfileReport(purchases2, title='Final Report',minimal=True)
final_profile.to_notebook_iframe()
purchases2['in_test'] = np.where(purchases2['time']>=purchases2['ts'], 1, 0)

purchases2.head()
ax = sns.boxplot(x="tag", y="time",data=purchases2,hue='in_test', palette="Set3")
purchases2[['user_uid','in_test']].groupby('user_uid').nunique().value_counts(normalize=True)
ax = sns.boxplot(x="tag", y="ts",data=purchases2, palette="Set3")
ts75 = np.quantile(purchases2.ts,0.75)

print(ts75)

purchases3 = purchases2[purchases2['ts']<ts75]
ax = sns.boxplot(x="tag", y="ts",data=purchases3, palette="Set3")
ax = sns.boxplot(x="tag", y="time",data=purchases3,hue='in_test', palette="Set3")
sns.boxplot(x="tag", y="price",data=purchases3,hue='in_test', palette="Set3")
ax = sns.distplot(purchases3['price'])
ax = sns.distplot(np.log(purchases3['price']))
purchases3[['consumption_mode','in_test','price']].groupby(['consumption_mode','in_test']).mean()
purchases3[['consumption_mode','in_test','user_uid']].groupby(['consumption_mode','in_test']).count()
print(purchases3[purchases3['in_test']==0].time.max()-purchases3[purchases3['in_test']==0].time.min())

print(purchases3[purchases3['in_test']==1].time.max()-purchases3[purchases3['in_test']==1].time.min())
df_control = purchases3[purchases3['tag']=='control']

print(df_control.shape)

print(df_control['in_test'].value_counts())

print(df_control[['user_uid','in_test']].groupby('user_uid').nunique().value_counts(normalize=True))

df_control.head()
sns.boxplot(x="consumption_mode", y="price",data=df_control,hue='in_test', palette="Set3")
print(df_control.user_uid.nunique())

print(df_control.price.nunique())
def get_bs_sample(data, n_samples):

    indexes = np.random.randint(0,len(data),(n_samples,len(data)))

    samples = np.array(data)

    samples = samples[indexes]

    return samples
def get_bs_intervals(stat,a):

    interval = np.percentile(stat,[100*a/2.,100*(1-a/2.)])

    return interval
np.random.seed(1234)

before_test_median = map(np.median,get_bs_sample(df_control[df_control['in_test']==0]['price'],1000))

in_test_median = map(np.median,get_bs_sample(df_control[df_control['in_test']==1]['price'],1000))

delta_median = map(lambda x: x[1]-x[0],zip(before_test_median,in_test_median))

get_bs_intervals(list(delta_median),0.05)
np.random.seed(1234)

before_test_median = map(np.mean,get_bs_sample(df_control[df_control['in_test']==0]['price'],1000))

in_test_median = map(np.mean,get_bs_sample(df_control[df_control['in_test']==1]['price'],1000))

delta_median = map(lambda x: x[1]-x[0],zip(before_test_median,in_test_median))

get_bs_intervals(list(delta_median),0.05)
sns.boxplot(x="in_test", y="price",data=df_control, palette="Set3")
for tag_i in purchases3.tag.unique():



    ## Выбираем группу

    print(f'Группа: {tag_i}')

    df_tmp = purchases3[purchases3['tag']==tag_i]

    

    ## Интервал для медианы

    before_test_median = map(np.median,get_bs_sample(df_tmp[df_tmp['in_test']==0]['price'],1000))

    in_test_median = map(np.median,get_bs_sample(df_tmp[df_tmp['in_test']==1]['price'],1000))

    delta_median = map(lambda x: x[1]-x[0],zip(before_test_median,in_test_median))

    delta_median_bs = get_bs_intervals(list(delta_median),0.05)

    print(f'Доверительный интервал для медианы: {delta_median_bs}')

    

    ## Интервал для среднего

    before_test_mean = map(np.mean,get_bs_sample(df_tmp[df_tmp['in_test']==0]['price'],1000))

    in_test_mean = map(np.mean,get_bs_sample(df_tmp[df_tmp['in_test']==1]['price'],1000))

    delta_mean = map(lambda x: x[1]-x[0],zip(before_test_mean,in_test_mean))

    delta_mean_bs = get_bs_intervals(list(delta_mean),0.05)

    print(f'Доверительный интервал для среднего: {delta_mean_bs}')
from scipy.stats import kruskal
df_in_test = purchases3[purchases3['in_test']==1] 

price_control = df_in_test[df_in_test['tag']=='control']['price']

price_test1 = df_in_test[df_in_test['tag']=='test1']['price']

price_test2 = df_in_test[df_in_test['tag']=='test2']['price']

price_test3 = df_in_test[df_in_test['tag']=='test3']['price']

price_test4 = df_in_test[df_in_test['tag']=='test4']['price']
kruskal(price_control,price_test1,price_test2,price_test3,price_test4)
users3.head()
users3.tag.value_counts(normalize=True)
users3[['tag','ts']].groupby('tag').agg(['mean','var'])
ax = sns.violinplot(x="tag", y="registration_time", data=users3)
ax = sns.violinplot(x="tag", y="conv_ts", data=users3)
users3['time_to_first_purchase']=users3['conv_ts']-users3['registration_time']
ax = sns.violinplot(x="tag", y="time_to_first_purchase", data=users3)
sns.distplot(users3[users3['tag']=='test3']['time_to_first_purchase'],kde=False,rug=True)
sns.distplot(users3[users3['tag']=='test1']['time_to_first_purchase'],kde=False,rug=True)
sns.distplot(users3[users3['tag']=='test2']['time_to_first_purchase'],kde=False,rug=True)
sns.distplot(users3[users3['tag']=='test4']['time_to_first_purchase'],kde=False,rug=True)
sns.distplot(users3[users3['tag']=='control']['time_to_first_purchase'],kde=False,rug=True)
purchases3_before = purchases3[purchases3['in_test']==0]

purchases3_before.head()
plt.style.use('seaborn')



test1 = purchases3_before[purchases3_before['tag']=='test1']['consumption_mode'].value_counts().to_frame().T

test2 = purchases3_before[purchases3_before['tag']=='test2']['consumption_mode'].value_counts().to_frame().T

test3 = purchases3_before[purchases3_before['tag']=='test4']['consumption_mode'].value_counts().to_frame().T

test4 = purchases3_before[purchases3_before['tag']=='test4']['consumption_mode'].value_counts().to_frame().T

control = purchases3_before[purchases3_before['tag']=='control']['consumption_mode'].value_counts().to_frame().T

total = purchases3_before['consumption_mode'].value_counts().to_frame().T



total = pd.concat([test1, test2,test3,test4,control,total])



fig, ax = plt.subplots(nrows=3, ncols=2)



labels = 'dto', 'subscription','rent'

colors = ['#008fd5', '#fc4f30','#06d6a0']

explode = (0, 0.1,0.1)



plt.title('Consumption mode weights')

plt.tight_layout()



ax[0,0].pie(total.iloc[[0]], startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', explode=explode, shadow=True)

ax[0,0].set_title('Test1 consumption_mode weights', fontweight='bold')

ax[0,1].pie(total.iloc[[1]],  startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', explode=explode, shadow=True)

ax[0,1].set_title('Test2 consumption_mode weights', fontweight='bold')

ax[1,0].pie(total.iloc[[2]], startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', explode=explode, shadow=True)

ax[1,0].set_title('Test3 consumption_mode weights', fontweight='bold')

ax[1,1].pie(total.iloc[[3]],  startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', explode=explode, shadow=True)

ax[1,1].set_title('Test4 consumption_mode weights', fontweight='bold')

ax[2,0].pie(total.iloc[[4]],  startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', explode=explode, shadow=True)

ax[2,0].set_title('Control consumption_mode weights', fontweight='bold')

ax[2,1].pie(total.iloc[[5]],  startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}, autopct='%1.f%%', explode=explode, shadow=True)

ax[2,1].set_title('Total consumption_mode weights', fontweight='bold')



fig.suptitle('Consumption mode weights', fontsize=20, y=1.07, fontweight='bold', x=0.37)

fig.set_figheight(10)

fig.set_figwidth(7)

fig.legend(loc='best', labels=labels, fontsize='medium')

fig.tight_layout()



plt.show()
purchases3_before[['tag','price']].groupby('tag').quantile([0.1,0.25,0.5,0.75,0.9])