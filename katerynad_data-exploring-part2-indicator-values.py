import pandas as pd

from pandas import Series,DataFrame

import numpy as np



# For Visualization

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns
df=pd.read_csv('../input/indicators_by_company.csv')
df.head()
years=['2011','2012','2013','2014','2015']
indicators=['Assets',

'LiabilitiesAndStockholdersEquity',

'StockholdersEquity',

'NetIncomeLoss',

'CashAndCashEquivalentsAtCarryingValue']
df_rtba=df.loc[df['indicator_id'].isin(indicators),['company_id','indicator_id','2011','2012','2013','2014','2015']]

df_rtba.head(10)
l_df=[]

for y in years:

    for c in indicators:

        d=list(df_rtba.loc[df_rtba['indicator_id']==c,y].dropna().describe())

        d.insert(0,y)

        d.insert(1,c)

        l_df.append(d)

df_ind_desc=DataFrame(l_df,columns=['Year','Indicator','count','mean','std','min','25%','50%','75%','max'])

df_ind_desc.head(20)
Assets=df_rtba.loc[df_rtba['indicator_id']=='Assets']

Assets=pd.melt(Assets, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

Assets.drop('indicator_id', axis=1, inplace=True)

Assets.dropna(inplace=True)

Assets.head()
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.countplot(x='year',data=Assets)

ax.set_title('Assets')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))





ax = sns.boxplot(x='year',y='value',data=Assets,

                 whis=np.inf)

sns.stripplot(x="year", y="value", data=Assets,

              jitter=True, size=3, color=".3", linewidth=0)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('Assets')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.barplot(x='year',y='value',data=Assets)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('Assets')
fig, ax = plt.subplots()

sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

for y in years:

    ax=sns.distplot(Assets.loc[Assets['year']==y,'value'],bins=10000,ax=ax, kde=False)

ax.set_xscale("log")

sns.despine(trim=True)

ax.set_title('Assets')
LiabilitiesAndStockholdersEquity=df_rtba.loc[df_rtba['indicator_id']=='LiabilitiesAndStockholdersEquity']

LiabilitiesAndStockholdersEquity=pd.melt(LiabilitiesAndStockholdersEquity, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

LiabilitiesAndStockholdersEquity.drop('indicator_id', axis=1, inplace=True)

LiabilitiesAndStockholdersEquity.dropna(inplace=True)

LiabilitiesAndStockholdersEquity.head()
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.countplot(x='year',data=LiabilitiesAndStockholdersEquity)

ax.set_title('LiabilitiesAndStockholdersEquity')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))





ax = sns.boxplot(x='year',y='value',data=LiabilitiesAndStockholdersEquity,

                 whis=np.inf)

sns.stripplot(x="year", y="value", data=LiabilitiesAndStockholdersEquity,

              jitter=True, size=3, color=".3", linewidth=0)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('LiabilitiesAndStockholdersEquity')

sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.barplot(x='year',y='value',data=LiabilitiesAndStockholdersEquity)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('LiabilitiesAndStockholdersEquity')
fig, ax = plt.subplots()

sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

for y in years:

    ax=sns.distplot(LiabilitiesAndStockholdersEquity.loc[LiabilitiesAndStockholdersEquity['year']==y,'value'],bins=10000,ax=ax, kde=False)

ax.set_xscale("log")

sns.despine(trim=True)

ax.set_title('LiabilitiesAndStockholdersEquity')
StockholdersEquity=df_rtba.loc[df_rtba['indicator_id']=='StockholdersEquity']

StockholdersEquity=pd.melt(StockholdersEquity, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

StockholdersEquity.drop('indicator_id', axis=1, inplace=True)

StockholdersEquity.dropna(inplace=True)

StockholdersEquity.head()
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.countplot(x='year',data=StockholdersEquity)

ax.set_title('StockholdersEquity')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))





ax = sns.boxplot(x='year',y='value',data=StockholdersEquity,

                 whis=np.inf)

sns.stripplot(x="year", y="value", data=StockholdersEquity,

              jitter=True, size=3, color=".3", linewidth=0)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('StockholdersEquity')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.barplot(x='year',y='value',data=StockholdersEquity)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('StockholdersEquity')
fig, ax = plt.subplots()

sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

for y in years:

    ax=sns.distplot(StockholdersEquity.loc[StockholdersEquity['year']==y,'value'],bins=1000,ax=ax, kde=False)

ax.set_xscale("log")

sns.despine(trim=True)

ax.set_title('StockholdersEquity')
NetIncomeLoss=df_rtba.loc[df_rtba['indicator_id']=='StockholdersEquity']

NetIncomeLoss=pd.melt(NetIncomeLoss, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

NetIncomeLoss.drop('indicator_id', axis=1, inplace=True)

NetIncomeLoss.dropna(inplace=True)

NetIncomeLoss.head()
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.countplot(x='year',data=NetIncomeLoss)

ax.set_title('NetIncomeLoss')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))





ax = sns.boxplot(x='year',y='value',data=NetIncomeLoss,

                 whis=np.inf)

sns.stripplot(x="year", y="value", data=NetIncomeLoss,

              jitter=True, size=3, color=".3", linewidth=0)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('NetIncomeLoss')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.barplot(x='year',y='value',data=NetIncomeLoss)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('NetIncomeLoss')
fig, ax = plt.subplots()

sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

for y in years:

    ax=sns.distplot(NetIncomeLoss.loc[NetIncomeLoss['year']==y,'value'],bins=1000,ax=ax, kde=False)

ax.set_xscale("log")

sns.despine(trim=True)

ax.set_title('NetIncomeLoss')
CashAndCashEquivalentsAtCarryingValue=df_rtba.loc[df_rtba['indicator_id']=='StockholdersEquity']

CashAndCashEquivalentsAtCarryingValue=pd.melt(CashAndCashEquivalentsAtCarryingValue, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

CashAndCashEquivalentsAtCarryingValue.drop('indicator_id', axis=1, inplace=True)

CashAndCashEquivalentsAtCarryingValue.dropna(inplace=True)

CashAndCashEquivalentsAtCarryingValue.head()
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.countplot(x='year',data=CashAndCashEquivalentsAtCarryingValue)

ax.set_title('CashAndCashEquivalentsAtCarryingValue')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))





ax = sns.boxplot(x='year',y='value',data=CashAndCashEquivalentsAtCarryingValue,

                 whis=np.inf)

sns.stripplot(x="year", y="value", data=CashAndCashEquivalentsAtCarryingValue,

              jitter=True, size=3, color=".3", linewidth=0)

ax.set_yscale("log")

sns.despine(trim=True)

ax.set_title('CashAndCashEquivalentsAtCarryingValue')
sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

ax=sns.barplot(x='year',y='value',data=CashAndCashEquivalentsAtCarryingValue)

ax.set_yscale("log")

sns.despine(trim=True)

fig, ax = plt.subplots()

sns.set(style="ticks", palette="muted", color_codes=True)

plt.figure(figsize=(12,8))

for y in years:

    ax=sns.distplot(CashAndCashEquivalentsAtCarryingValue.loc[CashAndCashEquivalentsAtCarryingValue['year']==y,'value'],bins=1000,ax=ax, kde=False)

ax.set_xscale("log")

sns.despine(trim=True)

ax.set_title('CashAndCashEquivalentsAtCarryingValue')