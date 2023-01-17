import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid')
data_df = pd.read_csv("../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv")
data_df.head()
data_df.describe()
data_df.info()
fig,ax = plt.subplots(figsize=(20,10))

sns.countplot(data_df['country'],hue=data_df['systemic_crisis'],ax=ax)

plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
systemic = data_df[['year','country', 'systemic_crisis', 'exch_usd', 'banking_crisis']]

systemic = systemic[(systemic['country'] == 'Central African Republic') | (systemic['country']=='Kenya') | (systemic['country']=='Zimbabwe') ]

plt.figure(figsize=(20,15))

count = 1



for country in systemic.country.unique():

    plt.subplot(len(systemic.country.unique()),1,count)

    subset = systemic[(systemic['country'] == country)]

    sns.lineplot(subset['year'],subset['systemic_crisis'],ci=None)

    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Systemic Crisis/Banking Crisis')

    plt.title(country)

    count+=1
plt.figure(figsize=(15,30))

count = 1

for country in data_df.country.unique():

    plt.subplot(len(data_df.country.unique()),1,count)

    count+=1

    sns.lineplot(data_df[data_df.country==country]['year'],data_df[data_df.country==country]['exch_usd'])

    plt.subplots_adjust(hspace=0.8)

    plt.xlabel('Years')

    plt.ylabel('Exchange Rates')

    plt.title(country)
fig,ax = plt.subplots(figsize=(20,10))

sns.countplot(data_df['country'],hue=data_df['domestic_debt_in_default'],ax=ax)

plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
sovereign = data_df[['year','country', 'domestic_debt_in_default', 'exch_usd', 'banking_crisis']]

sovereign = sovereign[(sovereign['country'] == 'Angola') | (sovereign['country']=='Zimbabwe') ]

plt.figure(figsize=(20,15))

count = 1



for country in sovereign.country.unique():

    plt.subplot(len(sovereign.country.unique()),1,count)

    subset = sovereign[(sovereign['country'] == country)]

    sns.lineplot(subset['year'],subset['domestic_debt_in_default'],ci=None)

    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Sovereign Domestic Debt Defaults/Banking Crisis')

    plt.title(country)

    count+=1
fig,ax = plt.subplots(figsize=(20,10))

sns.countplot(data_df['country'],hue=data_df['sovereign_external_debt_default'],ax=ax)

plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
sovereign_ext = data_df[['year','country', 'sovereign_external_debt_default', 'exch_usd', 'banking_crisis']]

sovereign_ext = sovereign_ext[(sovereign_ext['country'] == 'Central African Republic') | (sovereign_ext['country'] == 'Ivory Coast') | (sovereign_ext['country']=='Zimbabwe') ]

plt.figure(figsize=(20,15))

count = 1



for country in sovereign_ext.country.unique():

    plt.subplot(len(sovereign_ext.country.unique()),1,count)

    subset = sovereign_ext[(sovereign_ext['country'] == country)]

    sns.lineplot(subset['year'],subset['sovereign_external_debt_default'],ci=None)

    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Sovereign Ext Debt Defaults/Banking Crisis')

    plt.title(country)

    count+=1
fig,ax = plt.subplots(figsize=(20,10))

sns.countplot(data_df['country'],hue=data_df['currency_crises'],ax=ax)

plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
curr = data_df[['year','country', 'currency_crises', 'exch_usd', 'banking_crisis']]

curr = curr[(curr['country'] == 'Angola') | (curr['country'] == 'Zambia') | (curr['country']=='Zimbabwe') ]

curr = curr.replace(to_replace=2, value=1, regex=False)



plt.figure(figsize=(20,15))

count = 1



for country in curr.country.unique():

    plt.subplot(len(curr.country.unique()),1,count)

    subset = curr[(curr['country'] == country)]

    sns.lineplot(subset['year'],subset['currency_crises'],ci=None)

    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Currency Crisis/Banking Crisis')

    plt.title(country)

    count+=1
fig,ax = plt.subplots(figsize=(20,10))

sns.countplot(data_df['country'],hue=data_df['inflation_crises'],ax=ax)

plt.xlabel('Countries')

plt.ylabel('Counts')

plt.xticks(rotation=45)
infla = data_df[['year','country', 'inflation_crises', 'inflation_annual_cpi', 'banking_crisis']]

infla = infla[(infla['country'] == 'Angola') | (infla['country'] == 'Zambia') | (infla['country']=='Zimbabwe') ]

infla = infla.replace(to_replace=2, value=1, regex=False)



plt.figure(figsize=(20,15))

count = 1



for country in infla.country.unique():

    plt.subplot(len(infla.country.unique()),1,count)

    subset = infla[(infla['country'] == country)]

    sns.lineplot(subset['year'],subset['inflation_crises'],ci=None)

    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Inflation Crisis/Banking Crisis')

    plt.title(country)

    count+=1
plt.figure(figsize=(20,15))

count = 1



for country in infla.country.unique():

    plt.subplot(len(infla.country.unique()),1,count)

    subset = infla[(infla['country'] == country)]

    sns.lineplot(subset[subset.country==country]['year'], subset[subset.country==country]['inflation_annual_cpi'])

    plt.subplots_adjust(hspace=0.6)

    plt.xlabel('Years')

    plt.ylabel('Annual CPI')

    plt.title(country)

    count+=1