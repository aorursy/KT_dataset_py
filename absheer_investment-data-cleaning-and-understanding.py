#Importing libries

import numpy as np

import pandas as pd

import string

import matplotlib.pyplot as plt

import seaborn as sns
# reading data files

# using encoding = "ISO-8859-1" to avoid pandas encoding error



companies = pd.read_csv("../input/investment-analysis/companies.txt", sep="\t", encoding = "ISO-8859-1")

rounds = pd.read_csv("../input/investment-analysis/rounds2.csv", encoding = "ISO-8859-1")
#Overview of the data in companies file

companies.info()
# Checking the shape of the DataFrames

print("Companies File: ", companies.shape)

print("Rounds File", rounds.shape)
#Find Primary Key in both data Frames.

print("Shape of companies: ", companies.shape)

for col in companies.columns:

    print("Unique values in {}:{}".format(col, companies[col].nunique()))
print("Shape of Rounds: ", rounds.shape)

for col in rounds.columns:

    print("Unique value in column: {} is {}".format(col, rounds[col].nunique()))
#Also, let's convert all the entries to lowercase (or uppercase) for uniformity.

# converting all permalinks to lowercase

companies['permalink'] = companies['permalink'].str.lower()

rounds['company_permalink'] = rounds['company_permalink'].str.lower()
# identify the unique number of permalinks in companies

len(companies.permalink.unique())
# look at unique company names in rounds master

# note that the column name in rounds file is different (company_permalink)

len(rounds.company_permalink.unique())
# will use this columns to find the mismatch in it, 

# Present in companies but not in rounds

set(companies.permalink) - set(rounds.company_permalink )
# companies present in rounds master but not in companies master

set(rounds.company_permalink ) - set(companies.permalink)
# companies present in companies master but not in rounds master

companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]
# companies present in rounds file but not in (~) companies file

rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]
import chardet



rawdata = open('../input/investment-analysis/rounds2.csv', 'rb').read()

result = chardet.detect(rawdata)

charenc = result['encoding']

print(charenc)



# print(result)
# trying different encodings

# encoding="cp1254" throws an error

# rounds_original = pd.read_csv("rounds2.csv", encoding="cp1254")

# rounds_original.iloc[[29597, 31863, 45176], :]
# remove encoding from companies master

companies['permalink'] = companies.permalink.str.encode('utf-8').str.decode('ascii', 'ignore')



# remove encoding from rounds master

rounds['company_permalink'] = rounds.company_permalink.str.encode('utf-8').str.decode('ascii', 'ignore')
# missing values in companies master

companies.isnull().sum()
# missing values in rounds master

rounds.isnull().sum()
# merging the two masters

master = pd.merge(companies, rounds, how="inner", left_on="permalink", right_on="company_permalink")

master.head()
# print column names

master.columns
# removing redundant columns

master =  master.drop(['company_permalink'], axis=1) 
# column-wise missing values 

master.isnull().sum()
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

country_codes.value_counts().head(10)
# displaying frequencies of each category

country_codes.value_counts().tail(10)
# viewing fractions of counts of country_codes

100*(master['country_code'].value_counts()/len(master.index)).head(10)
# viewing fractions of counts of country_codes

100*(master['country_code'].value_counts()/len(master.index)).tail(10)
# removing rows with missing country_codes

master = master[~pd.isnull(master['country_code'])]



# look at missing values

round(100*(master.isnull().sum()/len(master.index)), 2)
# removing rows with missing category_list values

master = master[~pd.isnull(master['category_list'])]



# look at missing values

round(100*(master.isnull().sum()/len(master.index)), 2)
# first, let's filter the master so it only contains the four specified funding types

master = master[(master.funding_round_type == "venture") | 

        (master.funding_round_type == "angel") | 

        (master.funding_round_type == "seed") | 

        (master.funding_round_type == "private_equity") ]



# distribution of raised_amount_usd

sns.boxplot(y=master['raised_amount_usd'])

plt.yscale('log')

plt.show()
# summary metrics

master['raised_amount_usd'].describe()
# comparing summary stats across four categories

sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=master)

plt.yscale('log')

plt.show()
# compare the mean and median values across categories

master.pivot_table(values='raised_amount_usd', columns='funding_round_type', aggfunc=[np.median, np.mean])
# compare the median investment amount across the types

master.groupby('funding_round_type')['raised_amount_usd'].median().sort_values(ascending=False)
# filter the master for private equity type investments

master = master[master.funding_round_type=="venture"]



# group by country codes and compare the total funding amounts

country_wise_total = master.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False)

country_wise_total.head()
# top 9 countries

top_9_countries = country_wise_total[:9]

top_9_countries
# filtering for the top three countries

master = master[(master.country_code=='USA') | (master.country_code=='GBR') | (master.country_code=='IND')]
# filtered master has about 38800 observations

master.info()
# boxplot to see distributions of funding amount across countries

plt.figure(figsize=(8, 5))

sns.boxplot(x='country_code', y='raised_amount_usd', data=master)

plt.yscale('log')

plt.show()
# extracting the main category

master.loc[:, 'main_category'] = master['category_list'].apply(lambda x: x.split("|")[0])

master.head(2)
# drop the category_list column

master = master.drop('category_list', axis=1)

master.head(2)
# read mapping file

mapping = pd.read_csv("../input/investment-analysis/mapping.csv", sep=",")

mapping.head()
# missing values in mapping file

mapping.isnull().sum()
# remove the row with missing values

mapping = mapping[~pd.isnull(mapping['category_list'])]

mapping.isnull().sum()
# converting common columns to lowercase

mapping['category_list'] = mapping['category_list'].str.lower()

master['main_category'] = master['main_category'].str.lower()
# look at heads

mapping.head(2)
master.head(2)
mapping['category_list'][:10]
# values in main_category column in master which are not in the category_list column in mapping file

master[~master['main_category'].isin(mapping['category_list'])].head(10)
# values in the category_list column which are not in main_category column 

mapping.loc[mapping.category_list.str.contains('0')]
# replacing '0' with 'na'

mapping['category_list'] = mapping['category_list'].apply(lambda x: x.replace('0', 'na'))

mapping['category_list'].head(20)



#This can be Done Using Regex;

# mapping.loc[mapping['category_list'].str.match('.*\d+[\w\s]'),'category_list'] = mapping.category_list.str.replace('0','na')

# mapping.loc[mapping.category_list.str.contains('0')]
# merge the masters

master = pd.merge(master, mapping, how='inner', left_on='main_category', right_on='category_list')

master.head()
# let's drop the category_list column since it is the same as main_category

master = master.drop('category_list', axis=1)

master.head()
# look at the column types and names

master.info()
### help(pd.melt)
# store the value and id variables in two separate arrays



# store the value variables in one Series

value_vars = master.columns[9:18]



# take the setdiff() to get the rest of the variables

id_vars = np.setdiff1d(master.columns, value_vars)

# convert into long

long_master = pd.melt(master, 

        id_vars=list(id_vars), 

        value_vars=list(value_vars))



long_master.head()
# remove rows having value=0

long_master = long_master[long_master['value']==1]

long_master = long_master.drop('value', axis=1)
# look at the new master

long_master.head()

len(long_master)
# renaming the 'variable' column

long_master = long_master.rename(columns={'variable': 'sector'})
# summarising the sector-wise number and sum of venture investments across three countries



# first, let's also filter for investment range between 5 and 15m

master = long_master[(long_master['raised_amount_usd'] >= 5000000) & (long_master['raised_amount_usd'] <= 15000000)]

# groupby country, sector and compute the count and sum

master.groupby(['country_code', 'sector']).raised_amount_usd.agg(['count', 'sum'])
# plotting sector-wise count and sum of investments in the three countries

plt.figure(figsize=(16, 14))



plt.subplot(2, 1, 1)

p = sns.barplot(x='sector', y='raised_amount_usd', hue='country_code', data=master, estimator=np.sum)

p.set_xticklabels(p.get_xticklabels(),rotation=30)

plt.title('Total Invested Amount (USD)')



plt.subplot(2, 1, 2)

q = sns.countplot(x='sector', hue='country_code', data=master)

q.set_xticklabels(q.get_xticklabels(),rotation=30)

plt.title('Number of Investments')





plt.show()