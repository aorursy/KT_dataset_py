import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport



# reading data files

# using encoding = "ISO-8859-1" to avoid pandas encoding error

rounds = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/rounds2.csv", encoding = "ISO-8859-1")

companies = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/companies.txt", sep="\t", encoding = "ISO-8859-1")

profile = ProfileReport(rounds, title='Rounds Profiling Report', html={'style':{'full_width':True}})

profile
profile = ProfileReport(companies, title='Companies Profiling Report', html={'style':{'full_width':True}})

profile
# look at companies head

companies.head()
# identify the unique number of permalinks in companies

len(companies.permalink.unique())
# converting all permalinks to lowercase

companies['permalink'] = companies['permalink'].str.lower()

companies.head()

# look at unique values again

len(companies.permalink.unique())
# look at unique company names in rounds df

# note that the column name in rounds file is different (company_permalink)

len(rounds.company_permalink.unique())

# converting column to lowercase

rounds['company_permalink'] = rounds['company_permalink'].str.lower()

rounds.head()
# Look at unique values again

len(rounds.company_permalink.unique())
# companies present in rounds file but not in (~) companies file

rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]
# looking at the indices with weird characters

rounds_original = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/rounds2.csv", encoding = "ISO-8859-1")

rounds_original.iloc[[29597, 31863, 45176, 58473], :]
rounds['company_permalink'] = rounds.company_permalink.str.encode('utf-8').str.decode('ascii', 'ignore')

rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]
# Look at unique values again

len(rounds.company_permalink.unique())
# companies present in companies df but not in rounds df

companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]

# remove encoding from companies df

companies['permalink'] = companies.permalink.str.encode('utf-8').str.decode('ascii', 'ignore')

# companies present in companies df but not in rounds df

companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]

# quickly verify that there are 66368 unique companies in both

# and that only the same 66368 are present in both files



# unqiue values

print(len(companies.permalink.unique()))

print(len(rounds.company_permalink.unique()))



# present in rounds but not in companies

print(len(rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]))
# missing values in companies df

companies.isnull().sum()
# missing values in rounds df

rounds.isnull().sum()
# merging the two dfs

master = pd.merge(companies, rounds, how="inner", left_on="permalink", right_on="company_permalink")

master.head()
# print column names

master.columns
# removing redundant columns

master =  master.drop(['company_permalink'], axis=1) 
# look at columns after dropping

master.columns
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
# look at the master df info for number of rows etc.

master.info()
# after missing value treatment, approx 77% observations are retained

100*(len(master.index) / len(rounds.index))
# first, let's filter the df so it only contains the four specified funding types

df = master[(master.funding_round_type == "venture") | 

        (master.funding_round_type == "angel") | 

        (master.funding_round_type == "seed") | 

        (master.funding_round_type == "private_equity") ]
# distribution of raised_amount_usd

sns.boxplot(y=df['raised_amount_usd'])

plt.yscale('log')

plt.show()
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

print(country_wise_total)
# top 9 countries

top_9_countries = country_wise_total[:9]

top_9_countries
# filtering for the top three countries

df = df[(df.country_code=='USA') | (df.country_code=='GBR') | (df.country_code=='IND')]

df.head()
# filtered df has about 38800 observations

df.info()
# boxplot to see distributions of funding amount across countries

plt.figure(figsize=(10, 10))

sns.boxplot(x='country_code', y='raised_amount_usd', data=df)

plt.yscale('log')

plt.show()
# extracting the main category

df.loc[:, 'main_category'] = df['category_list'].apply(lambda x: x.split("|")[0])

df.head()
# drop the category_list column

df = df.drop('category_list', axis=1)

df.head()
# read mapping file

mapping = pd.read_csv("/kaggle/input/spark-fund-investment-analysis/datasets/mapping.csv", sep=",")

mapping.head()
# missing values in mapping file

mapping.isnull().sum()
# remove the row with missing values

mapping = mapping[~pd.isnull(mapping['category_list'])]

mapping.isnull().sum()
# converting common columns to lowercase

mapping['category_list'] = mapping['category_list'].str.lower()

df['main_category'] = df['main_category'].str.lower()
# look at heads

print(mapping.head())
print(df.head())
mapping['category_list']
# values in main_category column in df which are not in the category_list column in mapping file

df[~df['main_category'].isin(mapping['category_list'])]
# values in the category_list column which are not in main_category column 

mapping[~mapping['category_list'].isin(df['main_category'])]
# replacing '0' with 'na'

mapping['category_list'] = mapping['category_list'].apply(lambda x: x.replace('0', 'na'))

print(mapping['category_list'])
# merge the dfs

df = pd.merge(df, mapping, how='inner', left_on='main_category', right_on='category_list')

df.head()
# let's drop the category_list column since it is the same as main_category

df = df.drop('category_list', axis=1)

df.head()
# look at the column types and names

df.info()
# store the value and id variables in two separate arrays



# store the value variables in one Series

value_vars = df.columns[9:18]



# take the setdiff() to get the rest of the variables

id_vars = np.setdiff1d(df.columns, value_vars)



print(value_vars, "\n")

print(id_vars)
# convert into long

long_df = pd.melt(df, 

        id_vars=list(id_vars), 

        value_vars=list(value_vars))



long_df.head()
# remove rows having value=0

long_df = long_df[long_df['value']==1]

long_df = long_df.drop('value', axis=1)
# look at the new df

long_df.head()

len(long_df)
# renaming the 'variable' column

long_df = long_df.rename(columns={'variable': 'sector'})
long_df.info()
# summarising the sector-wise number and sum of venture investments across three countries



# first, let's also filter for investment range between 5 and 15m

df = long_df[(long_df['raised_amount_usd'] >= 5000000) & (long_df['raised_amount_usd'] <= 15000000)]

# groupby country, sector and compute the count and sum

df.groupby(['country_code', 'sector']).raised_amount_usd.agg(['count', 'sum'])
# plotting sector-wise count and sum of investments in the three countries

plt.figure(figsize=(16, 14))



plt.subplot(2, 1, 1)

p = sns.barplot(x='sector', y='raised_amount_usd', hue='country_code', data=df, estimator=np.sum)

p.set_xticklabels(p.get_xticklabels(),rotation=30)

plt.title('Total Invested Amount (USD)')



plt.subplot(2, 1, 2)

q = sns.countplot(x='sector', hue='country_code', data=df)

q.set_xticklabels(q.get_xticklabels(),rotation=30)

plt.title('Number of Investments')





plt.show()