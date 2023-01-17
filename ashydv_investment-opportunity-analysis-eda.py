# Supress Warnings

import warnings

warnings.filterwarnings('ignore')

# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Setting working directory to required location

import os

print(os.listdir("../input"))
# reading data files

# using encoding = "ISO-8859-1" to avoid pandas encoding error

rounds = pd.DataFrame(pd.read_csv( "../input/rounds2.csv", encoding = "LATIN-1"))

rounds.head()
companies = pd.DataFrame(pd.read_csv("../input/companies.txt", sep="\t", encoding = "ISO-8859-1"))

companies.head()
companies.shape
companies.info()
companies.describe()
# inspect the structure 

rounds.shape
rounds.info()
rounds.describe()
# converting all permalinks to lowercase

companies['permalink'] = companies['permalink'].str.lower()

len(companies.permalink.unique())
# converting column to lowercase

rounds['company_permalink'] = rounds['company_permalink'].str.lower()

len(rounds.company_permalink.unique())
# companies present in companies df but not in rounds df

companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]
# Thus, the companies df also contains special characters. Let's treat those as well.
# remove encoding from companies and rounds df

companies['permalink'] = companies.permalink.str.encode('utf-8').str.decode('ascii', 'ignore')

companies['name'] = companies.name.str.encode('utf-8').str.decode('ascii', 'ignore')

rounds['company_permalink'] = rounds.company_permalink.str.encode('utf-8').str.decode('ascii', 'ignore')
# companies present in companies df but not in rounds df

companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]
# Look at unique values again

len(rounds.company_permalink.unique())
# companies present in companies df but not in rounds df

companies[~companies['permalink'].isin(rounds['company_permalink'])]
# quickly verify that there are 66368 unique companies in both

# and that only the same 66368 are present in both files



# unqiue values

print(len(companies.permalink.unique()))

print(len(rounds.company_permalink.unique()))



# present in rounds but not in companies

print(len(rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]))

print(len(companies[~companies['permalink'].isin(rounds['company_permalink'])]))
# missing values in companies df

companies.isnull().sum()
# missing values in rounds df

rounds.isnull().sum()
# merging the two dfs

master = pd.merge(companies, rounds, how="inner", left_on="permalink", right_on="company_permalink")

master.head()
# removing redundant columns

master =  master.drop(['company_permalink'], axis=1) 
# summing up the missing values (column-wise) and displaying fraction of NaNs

round(100*(master.isnull().sum()/len(master.index)), 2)
# dropping columns 

master = master.drop(['funding_round_code', 'homepage_url', 'founded_at', 'state_code', 'region', 'city'], axis=1)

master.head()
# summing up the missing values (column-wise) and displaying fraction of NaNs

round(100*(master.isnull().sum()/len(master.index)), 2)
# summary stats of raised_amount_usd

master['raised_amount_usd'].describe()
# removing NaNs in raised_amount_usd

master = master[~np.isnan(master['raised_amount_usd'])]

round(100*(master.isnull().sum()/len(master.index)), 2)
country_codes = master['country_code'].astype('category')
# displaying frequencies of each category

country_codes.value_counts()
# viewing fractions of counts of country_codes

100*(master['country_code'].value_counts()/len(master.index))
# removing rows with missing country_codes

master = master[~pd.isnull(master['country_code'])]



# look at missing values

round(100*(master.isnull().sum()/len(master.index)), 2)
# removing rows with missing category_list values

master = master[~pd.isnull(master['category_list'])]



# look at missing values

round(100*(master.isnull().sum()/len(master.index)), 2)
master.info()
# Now the data looks nice and clean, let's proceed with the analysis.
# first, let's filter the df so it only contains the four specified funding types

df = master[(master.funding_round_type == "venture") | 

            (master.funding_round_type == "angel") | 

            (master.funding_round_type == "seed") | 

            (master.funding_round_type == "private_equity") ]

df.head()
# distribution of raised_amount_usd

plt1 = sns.boxplot(y=df['raised_amount_usd'])

plt.yscale('log')

plt1.set(ylabel = 'Funding ($)')

plt.tight_layout()

plt.show()
# First let's convert funding raised in million USD

df['raised_amount_usd'] = round(df['raised_amount_usd']/1000000,2)
# summary metrics

df['raised_amount_usd'].describe()
# comparing summary stats across four categories

sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=df)

plt.yscale('log')

plt.show()
# compare the mean and median values across categories

df.pivot_table(values='raised_amount_usd', columns='funding_round_type', aggfunc=[np.median, np.mean])
# compare the median investment amount across the types

df.groupby('funding_round_type')['raised_amount_usd'].median().sort_values(ascending=False)
# filter the df for private equity type investments

df = df[df.funding_round_type=="venture"]



# group by country codes and compare the total funding amounts

country_wise_total = df.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False)

country_wise_total[:9]
# filtering for the top three countries

df = df[(df.country_code=='USA') | (df.country_code=='GBR') | (df.country_code=='IND')]

df.head()
# boxplot to see distributions of funding amount across countries

plt.figure(figsize=(10, 10))

sns.boxplot(x='country_code', y='raised_amount_usd', data=df)

plt.yscale('log')

plt.show()
df["category_list"] = df["category_list"].str.split("|").str.get(0)

df.head()
mapping_table = pd.DataFrame(pd.read_csv( "../input/mapping.csv",))

mapping_table.head()
# Code for a merged data frame with each primary sector mapped to its main sector

# (the primary sector should be present in a separate column).

long_map = pd.melt(mapping_table, id_vars=['category_list'], var_name='main_sector')

long_map = long_map[long_map['value']==1]

long_map = long_map.drop('value',1)

long_map.head()
df = pd.merge(df, long_map, on = 'category_list' , how = 'inner')

df.head()
df.info()
# summarising the sector-wise number and sum of venture investments across three countries



# first, let's also filter for investment range between 5 and 15m

df = df[(df['raised_amount_usd'] >= 5) & (df['raised_amount_usd'] <= 15)]

df.head()
# First english speaking company 'USA' for funding type venture

D1 = df[df.country_code == 'USA']

# Second english speaking company 'Great Britain' for funding type venture

D2 = df[df.country_code == 'GBR']

# Third english speaking company 'India' for funding type venture

D3 = df[df.country_code == 'IND']
# groupby country, sector and compute the count and sum

df.groupby(['country_code', 'main_sector']).raised_amount_usd.agg(['count', 'sum'])
# plotting sector-wise count and sum of investments in the three countries

plt.figure(figsize=(16, 14))



plt.subplot(2, 1, 1)

p = sns.barplot(x='main_sector', y='raised_amount_usd', hue='country_code', data=df, estimator=np.sum)

p.set_xticklabels(p.get_xticklabels(),rotation=30)

plt.title('Total Invested Amount (USD)')



plt.subplot(2, 1, 2)

q = sns.countplot(x='main_sector', hue='country_code', data=df)

q.set_xticklabels(q.get_xticklabels(),rotation=30)

plt.title('Number of Investments')



plt.tight_layout()

plt.show()