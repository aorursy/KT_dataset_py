import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns', None)
data = pd.read_csv('../input/states_all_extended.csv')
data.head()
data.shape
data.describe()
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1]

print(len(vars_with_na))
vars_with_na
dict_missing = { var: np.round(data[var].isnull().mean()*100, 3) for var in vars_with_na}
dict_missing
import collections

sorted_dict = sorted(dict_missing.items(), key=lambda kv: kv[1], reverse=True)
sorted_dict
# create a dataframe of missing values

missings_df = pd.DataFrame.from_dict(sorted_dict)

missings_df.columns = ['columns', 'Percent missing']

missings_df.head()
missings_df.shape
d = missings_df.iloc[:50]
d
d = missings_df.iloc[:50]

plt.figure(figsize = [20, 10]);

g = sns.barplot(x="columns", y="Percent missing", data=d)

g.set_xticklabels(g.get_xticklabels(), rotation=90);
revenues = data[['YEAR','STATE','TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]
revenues
revenues_millions = revenues[['TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]/1000000
revenues_millions
# Create a figure and axes

fig, ax = plt.subplots(2, 2, figsize=(20, 10))



# plot the total revenue

ax[0, 0].hist(revenues_millions.TOTAL_REVENUE.dropna(), bins=50)

ax[0, 0].set_title('Total revenue in Millions')

ax[0, 0].set_xlabel('Revenue in Millions')

ax[0, 0].set_ylabel('Count')



# plot the federal revenue

ax[0, 1].hist(revenues_millions.FEDERAL_REVENUE.dropna(), bins=50)

ax[0, 1].set_title('Federal revenue in Millions')

ax[0, 1].set_xlabel('Revenue in Millions')

ax[0, 1].set_ylabel('Count')



# plot the state revenue

ax[1, 0].hist(revenues_millions.STATE_REVENUE.dropna(), bins=50)

ax[1, 0].set_title('State revenue in Millions')

ax[1, 0].set_xlabel('Revenue in Millions')

ax[1, 0].set_ylabel('Count')



# plot the local revenue

ax[1, 1].hist(revenues_millions.LOCAL_REVENUE.dropna(), bins=50)

ax[1, 1].set_title('Local revenue in Millions')

ax[1, 1].set_xlabel('Revenue in Millions')

ax[1, 1].set_ylabel('Count')

base_color = sns.color_palette()[2]

plt.figure(figsize = [10, 10])

plt.title('Revenue in Millions')

dfm = revenues_millions.melt(var_name='columns')

sns.violinplot(data = dfm, y='columns', x='value', color=base_color, inner = 'quartile')
base_color = sns.color_palette()[3]

plt.figure(figsize = [10, 10])

plt.title('Revenue in Millions')

dfm = revenues_millions.melt(var_name='columns')

sns.boxplot(data = dfm, y='columns', x='value', color=base_color)
dfm = pd.melt(revenues, id_vars =['YEAR'], value_vars =['TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE'])

dfm.columns = ['Year', 'Revenue_type', 'Dollar_Amount']

dfm.head()

dfm.Dollar_Amount = dfm.Dollar_Amount/1000000

dfm.head()

plt.figure(figsize = [20, 10])

plt.title('Revenue in millions')

sns.lineplot(x='Year', y='Dollar_Amount', hue='Revenue_type' , data=dfm, ci=None)
plt.figure(figsize = [20, 10])

plt.title('Average total revenue in Millions')

(revenues.groupby('YEAR')['TOTAL_REVENUE'].mean()/1000000).plot.bar()
total_rev = pd.concat([revenues['TOTAL_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="TOTAL_REVENUE", data=total_rev)

plt.ylabel('Total revenue in millions')

plt.title('Annual total revenue')
plt.figure(figsize = [20, 10])

plt.title('Average federal revenue in Millions')

(revenues.groupby('YEAR')['FEDERAL_REVENUE'].mean()/1000000).plot.bar()
total_rev = pd.concat([revenues['FEDERAL_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="FEDERAL_REVENUE", data=total_rev)

plt.xlabel('Federal revenue in millions')

plt.title('Annual Federal revenue')
plt.figure(figsize = [20, 10])

plt.title('Average state revenue in Millions')

(revenues.groupby('YEAR')['STATE_REVENUE'].mean()/1000000).plot.bar()
total_rev = pd.concat([revenues['STATE_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="STATE_REVENUE", data=total_rev)

plt.xlabel('State revenue in millions')

plt.title('Annual State revenue')
plt.figure(figsize = [20, 10])

plt.title('Average local revenue in Millions')

(revenues.groupby('YEAR')['LOCAL_REVENUE'].mean()/1000000).plot.bar()

total_rev = pd.concat([revenues['LOCAL_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="LOCAL_REVENUE", data=total_rev)

plt.xlabel('Local revenue in millions')

plt.title('Annual Local revenue')
rev_data = revenues.groupby('STATE')['TOTAL_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('TOTAL_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='TOTAL_REVENUE', figsize=(10, 25))

plt.xlabel('Average Total revenue in thousands')
total_rev = pd.concat([revenues['TOTAL_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="TOTAL_REVENUE", data=total_rev)

plt.xlabel('Total revenue in millions')

plt.title('Annual total revenue')
rev_data = revenues.groupby('STATE')['FEDERAL_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('FEDERAL_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='FEDERAL_REVENUE', figsize=(10, 25))

plt.xlabel('Average federal revenue in thousands')
total_rev = pd.concat([revenues['FEDERAL_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="FEDERAL_REVENUE", data=total_rev)

plt.xlabel('Federal revenue in millions')

plt.title('Annual Federal revenue')
rev_data = revenues.groupby('STATE')['STATE_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('STATE_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='STATE_REVENUE', figsize=(10, 25))

plt.xlabel('Average state revenue in thousands')
total_rev = pd.concat([revenues['STATE_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="STATE_REVENUE", data=total_rev)

plt.xlabel('State revenue in millions')

plt.title('Annual State revenue')
rev_data = revenues.groupby('STATE')['LOCAL_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('LOCAL_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='LOCAL_REVENUE', figsize=(10, 25))

plt.xlabel('Average local revenue in thousands')
total_rev = pd.concat([revenues['LOCAL_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="LOCAL_REVENUE", data=total_rev)

plt.xlabel('Local revenue in millions')

plt.title('Annual State revenue')