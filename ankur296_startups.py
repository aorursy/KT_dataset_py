import numpy as np

import seaborn as sns

sns.set_style('darkgrid')

import datetime

import pandas as pd

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import re

from string import Template

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def plot(xlabel, ylabel, title,  y_lim, y_init = 0, y_interval=10):

    """custom func to prevent repetition while plotting"""

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    plt.yticks(range(y_init, y_lim + y_interval, y_interval))

    plt.show()

    

def plot_sns(g, xlabel="", ylabel="", title="", y = 1.07 , y_lim=100, y_init = 0, y_interval=10, rot = 90,face_grid=True):

    if face_grid is True:

        g.fig.suptitle(title, y = y)

    else:

        g.set(title = title)

    g.set(xlabel = xlabel, ylabel = ylabel)

    plt.yticks(range(y_init, y_lim + y_interval, y_interval))

    plt.xticks(rotation=rot)

    plt.show()
df = pd.read_csv('/kaggle/input/indian-startups-funding-weekly-dataset-20162020/fundingdata.csv',encoding= 'unicode_escape',parse_dates=['Start Date','End Date'])

df.head(10)
print('='*20 + 'SHAPE' + '='*20)

print(df.shape)

print('\n\n')

print('='*20 + 'INFO' + '='*20)

df.info()

print('\n\n')

print('='*20 + 'DESCRIBE' + '='*20)

print(df.describe())

print('\n\n')

print('='*20 + 'VALUE COUNTS' + '='*20)

print(df["HF-Startup name"].value_counts())

print('\n\n')

print('='*20 + 'NA check' + '='*20)

print(df.isna().any())

print(df.isna().sum())

print('\n\n')

print('='*20 + 'UNIQUE' + '='*20)

print(f'Companies = {df["HF-Startup name"].unique()}')

df['Start Date'] < df['End Date']
df.columns = ['start_date','end_date', 'total_funding_usd', 'Name', 'highest_funding_usd', 'total_deals','undisclosed_deals','year']

df = df.rename(columns = {

    "Start Date": "start_date",

    "End Date": "end_date",

    "Total Funding Amount(in USD)": "total_funding_usd",

    "HF-Startup name": "Name",

    "Highest Funding": "highest_funding_usd",

    "Total Deals": "total_deals",

    "Undisclosed Deals": "undisclosed_deals",

    "Year" : "year"

})

df.columns

df.highest_funding_usd = df.highest_funding_usd/1000000

df.total_funding_usd = df.total_funding_usd/1000000

df = df.rename(columns = {

    "total_funding_usd": "total_funding_mill_usd",

    "highest_funding_usd": "highest_funding_mill_usd",

})

df.columns
df.Name = df.Name.str.strip()

df.Name = df.Name.str.upper()

df.replace({'POLICY BAZAAR':'POLICYBAZAAR'}, inplace=True)



for row in df.itertuples():

    if row.start_date >= row.end_date:

        temp = row.start_date

        df.at[row.Index, 'start_date'] = row.end_date

        df.at[row.Index, 'end_date'] = temp
df['duration'] = (df.end_date - df.start_date).dt.days

df.columns
plt.figure(figsize=(18,5))

interesting_companies = ['OYO', 'BYJU\'S', 'OLA','SWIGGY']

yearly_funding = df[df.Name.isin(interesting_companies)].groupby(['Name','year'])[['total_funding_mill_usd']].agg(sum).reset_index()

# print(yearly_funding)

y_lim = int(yearly_funding.total_funding_mill_usd.max())

plt.xticks([2016,2017,2018,2019,2020])

g = sns.lineplot(x="year", y = "total_funding_mill_usd", data = yearly_funding, hue='Name')

plot_sns(g,ylabel = 'Million USD', title = 'Fundings received', y_lim = y_lim, y_interval=200,face_grid=False,rot=0)



plt.figure(figsize=(12,8))

interesting_companies = ['OYO', 'BYJU\'S', 'OLA','SWIGGY']

sns.heatmap(pd.crosstab( df[df.total_funding_mill_usd >= 380]['Name'], df.year, values = round(df.total_funding_mill_usd/1000,2), aggfunc = 'mean'), annot=True,cmap='YlGnBu', linewidths=3)

plt.title('Fundings in $Billion over the years')

plt.ylabel('')

plt.xlabel('')

plt.show()



plt.figure(figsize=(18,5))

funding_per_year = df.groupby(['year'])[['total_funding_mill_usd']].agg(sum).reset_index()

funding_per_year['total_funding_bn_usd'] = round(funding_per_year['total_funding_mill_usd']/1000 , 1)

# print(funding_per_year)

y_lim = int(funding_per_year.total_funding_bn_usd.max())

plt.xticks([2016,2017,2018,2019,2020])

g = sns.lineplot(x="year", y = "total_funding_bn_usd", data = funding_per_year)

plot_sns(g,ylabel = 'Billion USD', title = 'Total Fundings (Till June 2020)', y_lim = y_lim, y_interval=2,face_grid=False,rot=0)



df_highest_funding_round = df[['Name', 'highest_funding_mill_usd']].nlargest(20,'highest_funding_mill_usd').set_index('Name')

# print(df_highest_funding_round)

y_lim = int(df_highest_funding_round.highest_funding_mill_usd.max())

df_highest_funding_round.plot(kind='bar',figsize=(18,8))

plt.legend('',frameon=False)

plot(xlabel = '', ylabel = 'Million USD', title = 'Companies which received highest funding in a round', y_lim=y_lim, y_interval=200)



total_funding = df.groupby('Name').total_funding_mill_usd.agg(sum).nlargest(25).reset_index()

total_funding['total_funding_bn_usd'] = round(total_funding.total_funding_mill_usd/1000, 1)

total_funding = total_funding.drop(columns = 'total_funding_mill_usd')

# print(total_funding)

plt.figure(figsize=(18,5))



y_lim = int(total_funding.total_funding_bn_usd.max())

g = sns.barplot(x = total_funding.Name, y = total_funding.total_funding_bn_usd)

plot_sns(g,ylabel = 'Billion USD', title = 'Companies which received most funding', y_lim = y_lim, y_interval=1,face_grid=False)

total_deals = df.groupby('Name').total_deals.agg(sum).nlargest(20).reset_index()

plt.figure(figsize=(18,5))

# print(total_deals)

y_lim = int(total_deals.total_deals.max())

g = sns.barplot(x = total_deals.Name, y = total_deals.total_deals)

plot_sns(g,ylabel = '', title = 'Companies with most deals', y_lim = y_lim, y_interval=20,face_grid=False)

total_undisclosed_deals = df.groupby('Name').undisclosed_deals.agg(sum).nlargest(20).reset_index()

plt.figure(figsize=(18,5))

# print(total_undisclosed_deals)

y_lim = int(total_undisclosed_deals.undisclosed_deals.max())

g = sns.barplot(x = total_undisclosed_deals.Name, y = total_undisclosed_deals.undisclosed_deals)

plot_sns(g,ylabel = '', title = 'Companies with most undisclosed deals', y_lim = y_lim, y_interval=5,face_grid=False)

df_deals_total = df.groupby(['Name']).total_deals.agg(sum).nlargest(50).reset_index()

df_deals_undisclosed = df.groupby(['Name']).undisclosed_deals.agg(sum).nlargest(50).reset_index()



df_deals = pd.merge(df_deals_total, df_deals_undisclosed, on=['Name'], how='inner').set_index('Name').iloc[0:15]

# print(df_deals)

df_deals.plot(color=["SkyBlue","IndianRed"],kind='bar', grid=True, figsize=(18,5))

y_lim = df_deals[['total_deals']].values.max()

plot(xlabel = '', ylabel = '', title = '', y_lim = y_lim, y_interval = 15)



df_duration = df[df.duration > 180][['Name','duration']].sort_values(by='duration',ascending=False).set_index('Name')

# print(df_duration)

df_duration.plot(kind='bar', grid=True, figsize=(18,5))

y_lim = df_duration.duration.values.max()

plot(xlabel = '', ylabel = 'Days', title = 'Max duration of funding', y_lim = y_lim, y_interval = 50)
df2 = pd.read_excel('/kaggle/input/indian-startups-funding-weekly-dataset-20162020/Indian_Unicorns.xlsx')

df2.head(10)
print('==========================SHAPE==========================')

print(df2.shape)

print('\n\n')

print('==========================INFO==========================')

df2.info()

print('\n\n')

print('==========================DESCRIBE==========================')

print(df2.describe())

print('\n\n')

print('==========================VALUE COUNTS==========================')

print(df2.Company.value_counts())

print(df2.Sector.value_counts())

print(df2.Entry.value_counts())

print(df2.Location.value_counts())

print('\n\n')

print('==========================NA check==========================')

print(df2.isna().any())

print(df2.isna().sum())

print('\n\n')

print('==========================UNIQUE==========================')

print(f'Companies = {df2.Company.unique()}')

print('\n')

print(f'Sectors = {df2.Sector.unique()}')

print('\n')

print(f'Valuations = {df2["Valuation ($B)"].unique()}')

print('\n')

print(f'Entry = {df2.Entry.unique()}')

print('\n')

print(f'Locations = {df2.Location.unique()}')



# REMOVE SPACES FROM STRINGS

df2.Company = df2.Company.str.strip()

df2.Location = df2.Location.str.strip()

# CONVERT TO UPPER CASE

df2.Company = df2.Company.str.upper()

df2.Location = df2.Location.str.upper()

df2.Sector = df2.Sector.str.upper()

# DROP UNNECESSARY COLS/ RENAME COLS

df2 = df2.drop(columns='No.')

df2 = df2.rename(columns = {"Company": "Name","Valuation ($B)":'Valuation_in_Bn_USD'})

print(df2.head(5))

valuation = df2[['Name','Valuation_in_Bn_USD','Location']].nlargest(30,'Valuation_in_Bn_USD')

plt.figure(figsize=(18,5))

# print(valuation)

y_lim = int(valuation.Valuation_in_Bn_USD.max())

g = sns.barplot(x = valuation.Name, y = valuation.Valuation_in_Bn_USD,hue=valuation.Location,dodge=False)

plot_sns(g,ylabel = '', title = 'Valuation in Billion$ of Unicorns', y_lim = y_lim, y_interval=2,face_grid=False)

valuation = df2[['Name','Valuation_in_Bn_USD','Entry']].nlargest(29,'Valuation_in_Bn_USD')

plt.figure(figsize=(18,5))

# print(valuation)

y_lim = int(valuation.Valuation_in_Bn_USD.max())

g = sns.barplot(x = valuation.Name, y = valuation.Valuation_in_Bn_USD,hue=valuation.Entry,dodge=False)

plot_sns(g,ylabel = '', title = 'Valuation in Billion$ of Unicorns', y_lim = y_lim, y_interval=2,face_grid=False)

loc = df2.Location.value_counts()

plt.figure(figsize=(8,5))

g = sns.barplot(x = loc.index, y = loc)

plot_sns(g, title = 'Cities hosting most Unicorns',y_lim = loc.max(), y_interval = 2, face_grid=False)
year_vc = df2.Entry.value_counts()

plt.figure(figsize=(10,5))

g = sns.barplot(x = year_vc.index, y = year_vc)

plot_sns(g, title = 'Year-wise distribution of when Unicorns were setup',y_lim = year_vc.max(), y_interval = 1, face_grid=False)

sector = df2.Sector.value_counts()

plt.figure(figsize=(15,5))

g = sns.barplot(x = sector.index, y = sector)

plot_sns(g, title = 'Sectors which Unicorns operated in',y_lim = sector.max(), y_interval = 1, face_grid=False)

plt.figure(figsize=(18,5))

df2['Valuation_in_Mill_USD'] = df2['Valuation_in_Bn_USD']*1000

# print(df2['Valuation_in_Mill_USD'].sort_values(ascending=False))

g = sns.boxplot(data = df2,  x = 'Valuation_in_Mill_USD')

plot_sns(g, xlabel='Million$',title = 'Distribution of Unicorns\' valuation', y_lim=1,y_interval = 1, face_grid=False)
