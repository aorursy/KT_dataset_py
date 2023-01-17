# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csvo

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)

import re 

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
kick_strt=pd.read_csv('../input/ks-projects-201801.csv',parse_dates=True)

kick_strt.info()

kick_strt.head()

kick_strt.describe()
kick_strt.columns



kick_strt.isnull().sum()/kick_strt.shape[0]
kick_strt['usd pledged']=kick_strt['usd pledged'].fillna(kick_strt['usd pledged'].median())
kick_strt.head()

kick_strt=kick_strt[kick_strt.name.notnull()]
print('category_count',kick_strt.category.nunique())

print('currency_count',kick_strt.currency.nunique())

print('main_category _count',kick_strt.main_category.nunique())

print('state_count',kick_strt.state.nunique())

print('country-count',kick_strt.country.nunique())

y=kick_strt['main_category'].value_counts()

x=kick_strt['main_category'].unique()

plt.figure(figsize=(len(kick_strt['main_category'].unique()),10))

sns.barplot(x,y)

plt.title('Main Categories')

plt.show()

y=kick_strt['category'].value_counts()[0:15]

x=kick_strt['category'].unique()[0:15]

plt.figure(figsize=(20,10))

sns.barplot(x,y)

plt.title('Top 15 SubCategories')

plt.show()
kick_strt['category'].value_counts()[0:15]
kick_strt.groupby(['main_category','category']).count()
kick_strt.columns
fig, ax = plt.subplots(1,2, sharex='row',sharey='col',figsize=(20,10))

plt.suptitle('Comparing pledged-goal Vs pledged Real-Goal Real')

ax[0].hist(np.sqrt(kick_strt.pledged), bins = np.linspace(0,50),alpha=0.5,label='pledged')

ax[0].hist(np.sqrt(kick_strt.goal), bins=np.linspace(0,50), alpha=0.5,label='goal')

ax[0].legend()

ax[1].hist(np.sqrt(kick_strt.usd_pledged_real), bins=np.linspace(0,50), alpha=0.5, label='Pledged Real')

ax[1].hist(np.sqrt(kick_strt.usd_goal_real), bins=np.linspace(0,50), alpha=0.5,label='Goal Real')



plt.legend()



plt.show()

plt.figure(figsize=(7,7))

plt.pie(kick_strt.state.value_counts(),labels=['failed','success','canceled','undefined','live','suspended'],explode=(0.1,0.2,0.3,0.4,0.5,0.6),autopct='%1.1f%%',shadow=True, startangle=45,rotatelabels=80)

plt.show()
kick_strt.country=kick_strt.country.str.replace('N,0"','Unknown')

kick_strt.country.value_counts()
kick_strt[kick_strt['state']=='failed'].groupby('currency').size().reset_index(name='Fail_count').sort_values(by='Fail_count',ascending=False).head(3)

kick_strt[kick_strt['state']=='successful'].groupby('currency').size().reset_index(name='success_count').sort_values(by='success_count',ascending=False).head(3)

kick_strt[kick_strt['state']=='canceled'].groupby('currency').size().reset_index(name='canceled_count').sort_values(by='canceled_count',ascending=False).head(3)

kick_strt[kick_strt['state']=='suspended'].groupby('currency').size().reset_index(name='suspended_count').sort_values(by='suspended_count',ascending=False).head(3)

kick_strt[kick_strt['state']=='live'].groupby('currency').size().reset_index(name='live_count').sort_values(by='live_count',ascending=False).head(3)

kick_strt[kick_strt['state']=='undefined'].groupby('currency').size().reset_index(name='undefined_count').sort_values(by='undefined_count',ascending=False).head(3)

country_result={'Currency':['USD','GBP','EUR'],'fail':[152130,17394,10496],'success':[109379,12081,4137],'canceled':[28326,3763,2389],'suspended':[1216,178,147],'live':[1741,329,279],'undefined':[2570,436,201]}

country_result=pd.DataFrame(country_result)

print(country_result)



                             
new=pd.melt(country_result,id_vars='Currency',value_vars=['fail','success','canceled','suspended','live','undefined'])

plt.figure(figsize=(6,6))

fig, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(6,15))

#plt.figure(0)

ax1.set_title('USD')

ax1.pie(new.loc[new['Currency']=='USD','value'],autopct='%1.1f%%',labels=['fail','success','canceled','suspended','live','undefined'],pctdistance=1.7,explode=[0,0,0,0.5,0.6,0.7])

ax2.set_title('GBP')

ax2.pie(new.loc[new['Currency']=='GBP','value'],autopct='%1.1f%%',labels=['fail','success','canceled','suspended','live','undefined'],pctdistance=1.7,explode=[0,0,0,0.5,0.6,0.7])

ax3.set_title('EUR')

ax3.pie(new.loc[new['Currency']=='EUR','value'],autopct='%1.1f%%',labels=['fail','success','canceled','suspended','live','undefined'],pctdistance=1.7,explode=[0,0,0,0.5,0.6,0.7])

plt.show()


kick_strt['launched']=pd.to_datetime(kick_strt['launched'])

kick_strt['deadline']=pd.to_datetime(kick_strt['deadline'])

kick_strt['Days']=kick_strt['deadline'].dt.round('D')-kick_strt['launched'].dt.round('D')



kick_strt['Days'].head()

kick_strt.Days.quantile(q=np.arange(0,1.1,0.1))



launch_dt=kick_strt['launched'].dt.strftime('%Y-%m-%d')

top=launch_dt.value_counts().sort_values(ascending=False).head()

low=launch_dt.value_counts().sort_values(ascending=True).head()



fig, ax = plt.subplots(1,2,figsize=(20,4))

plt.suptitle('Comparing the top5 and low5 launch dates')

ax[0].set_title('Top')

sns.barplot(x=top.index,y=top.values,ax=ax[0])

ax[1].set_title('Low')

sns.barplot(x=low.index,y=low.values)





kick_strt[['usd pledged','goal','usd_pledged_real','usd_goal_real','launched']].set_index('launched').resample('Y').mean().iplot(kind='bar', xTitle='Date', yTitle='Average',

    title='Yearly Average pledges and goals')
kick_strt.pivot(columns='main_category', values='backers').iplot(

        kind='box',

        yTitle='backers',

        title='backers Distribution by main category')

kick_strt.pivot(columns='state', values='backers').iplot(

        kind='box',

        yTitle='backers',

        title='backers Distribution by state')
kick_strt.pivot(columns='country', values='backers').iplot(

       kind='box',

        yTitle='backers',

        title='backers Distribution by country')


#kick_strt.groupby('country')['backers'].sum().sort_values(ascending=False).head(10)

kick_strt.groupby('country')['backers'].sum().sort_values(ascending=False).head(10).plot.pie(explode=(0,0,0,0,0,0,0.2,0.3,0.4,0.5),pctdistance=1.3,autopct='%1.1f%%',figsize=(10,20))