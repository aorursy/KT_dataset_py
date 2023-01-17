# Fisrt let's import the data.

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



data = pd.read_csv('../input/military-expenditure-of-countries-19602019/Military Expenditure.csv')

data.head(5)
# The 'Indicator Name' are all in USD. It looks like the data is quite clean and don't need much cleaning process.

# I noticed there's different kinds of values in 'Type' columns.

data['Type'].value_counts()
data[data['Type']=='Regions Clubbed Geographically']

# These regions clubbed geographically data represents large areas in the world and might have covered some contries as well. We need to analyze it seperately from contries.
# We can find countries with zero spending or no data on millitary spending. 

data['Total_Spend'] = data.iloc[:, 4:].sum(axis=1)

data[data['Total_Spend'] == 0].iloc[:, [0,1,2,3,-1]]



# 55 countries or regions with no records or zero millitary spending.
# What are the top 10 highest millitary total spending countries?



data_countries = data[data['Type']=='Country']

data_top10_countries = data_countries[['Name', 'Total_Spend']].sort_values(by='Total_Spend', ascending=False).iloc[:10, :]

sns.barplot(y='Name', x='Total_Spend', data=data_top10_countries)

plt.xlabel('Total Millitary Spend from 1960-2018(100 Trillion)')

plt.title('Total Millitary Spend from 1960-2018 by Countries')
# It looks like USA spends quite a lot on millitary, which is probably even bigger than the sum of the others.

# What about the top 10 highest millitary spending regions？



data_regions_g = data[data['Type']=='Regions Clubbed Geographically'][data['Name']!='World']

data_top10_regions_g = data_regions_g[['Name', 'Total_Spend']].sort_values(by='Total_Spend', ascending=False)

sns.barplot(y='Name', x='Total_Spend', data=data_top10_regions_g)

plt.xlabel('Total Spend(100 Trillion)')

plt.title('Total Millitary Spend of 58 Years by Regions')
data_top10_countries_list = data_top10_countries.Name.to_list() # get the name list of top 10 countries

# Make a dataframe of country name, year and spend.

data_top10_countries_df = data[data['Name'].isin(data_top10_countries_list)].drop(['Code', 'Type', 'Indicator Name', 'Total_Spend'], axis=1)

data_top10_countries_df = data_top10_countries_df.set_index('Name').stack().reset_index()

data_top10_countries_df.columns = ['Name', 'Year', 'Spend']

data_top10_countries_df.sort_values(by='Year', inplace=True) # Sort the dataframe by year



# Plot the data

f,ax = plt.subplots(figsize=(20, 10))

plt.xticks(rotation=45)

plt.grid(alpha=0.6)

sns.lineplot(x='Year', y='Spend', hue='Name', sort=True, data=data_top10_countries_df)

plt.ylabel('Spend(100 billion USD)')

plt.title('Trends of Top 10 highest Countries on Millitary Spending')
# We'll then take a look at different regions's millatary spending trend as well, by making a dataframe of region's name, year and spend.



data_regions_g_df = data_regions_g.drop(['Code', 'Type', 'Indicator Name', 'Total_Spend'], axis=1)

data_regions_g_df = data_regions_g_df.set_index('Name').stack().reset_index()

data_regions_g_df.columns = ['Name', 'Year', 'Spend']

data_regions_g_df.sort_values(by='Year', inplace=True)



f,ax = plt.subplots(figsize=(20, 10))

plt.xticks(rotation=45)

plt.grid(alpha=0.6)

sns.lineplot(x='Year', y='Spend', hue='Name', sort=True, data=data_regions_g_df)

plt.ylabel('Spending(1 Hundred billion USD)')

plt.title('Trends of Regions Clubbed Geographically on Millitary Spending')
# If we calculate the total spending of the world by adding up each coutries, we can get a world level millitary trend by year.



data_countries_df = data[data['Type']=='Country'][data['Name']!='World'].drop(['Code', 'Type', 'Indicator Name', 'Total_Spend'], axis=1)

data_countries_total = data_countries_df.sum()[1:]

data_countries_total.plot(kind='bar', figsize=(15,7))

plt.xlabel('Year')

plt.ylabel('Spending(1 trillion USD)')

plt.title('Trends of World\'s Millitary Spending')