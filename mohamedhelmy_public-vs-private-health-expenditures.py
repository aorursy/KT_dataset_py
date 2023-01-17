import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# data source is https://www.kaggle.com/worldbank/world-development-indicators

data = pd.read_csv('../input/Indicators.csv')

print(data.shape)

countries = pd.read_csv('../input/Country.csv')

print(countries.shape)
# View data frame columns and header

data.head()
countries.head()
# Explore range of data

print(data['Year'].max())

print(data['Year'].min())
# As regions might be easier to explore than countries, it is better to find out regions from countries data frame

print(countries['Region'].unique())

print(len(countries['Region'].unique()))
# As income groups might be easier to explore than countries, it is better to find out income groups from countries data frame

print(countries['IncomeGroup'].unique())

print(len(countries['IncomeGroup'].unique()))
# find indicators that include health data (contain 'health' word)

health_filter = data['IndicatorName'].str.contains('health|hospital|physician', case = False)

health_df = data[health_filter]

#health_df.head()

health_indicators = health_df['IndicatorName'].unique().tolist()

health_indicators
# find indicators that include pollution or CO2 data (contain 'pollution',or 'CO2')

pollution_filter = data['IndicatorName'].str.contains('pollution|CO2', case = False)

pollution_df = data[pollution_filter]

#pollution_df.head()

pollution_indicators = pollution_df['IndicatorName'].unique().tolist()

pollution_indicators
# find indicators that include life, death, or mortality data (contain 'life', 'death', or 'mortality')

yvar_filter = data['IndicatorName'].str.contains('life|death|mortality|poverty', case = False)

yvar_df = data[yvar_filter]

#yvar_df.head()

yvar_indicators = yvar_df['IndicatorName'].unique().tolist()

yvar_indicators
# Find out if there is an entry for world as a whole

world_filter = data['CountryName'].str.contains('world', case = False)

world_df = data[world_filter]

#world_df.head()

list_world = world_df['CountryName'].unique()

list_world
# put all indicators to be used in one data frame

df = data[health_filter | yvar_filter | pollution_filter | world_filter]

df.tail()

#check amount of data filtered

df.shape
CanUS_filter = ['Canada', 'United States']

Ind1_filter = ['Health expenditure, private (% of GDP)', 'Health expenditure, public (% of GDP)']

df_1 = df.loc[(df['CountryName'].isin(CanUS_filter)) & (df['IndicatorName'].isin(Ind1_filter))]

df_1.head()
df_2 = pd.pivot_table(df_1, values = ['Value'], index = ['CountryName', 'Year'], columns = ['IndicatorName'])

df_2
#df_2.dtypes

#df_2.loc['Canada', ('Value', 'Health expenditure, private (% of GDP)')]

#df_2.loc['Canada', ('Value', 'Health expenditure, private (% of GDP)')].plot(kind = 'line')

#df_2.loc['Canada', ('Value', 'Health expenditure, public (% of GDP)')].plot(kind = 'line')
fig1 = df_2.unstack(level=0).plot(kind = 'line', figsize = (12, 8))

plt.ylabel('Health Expenditure, % GDP', size = 'x-large')

fontdict = {'fontsize': 18, 'fontweight': 'normal'}

plt.title('Public vs. Private Health Expenditures in Canada and the United States', fontdict = fontdict)

years = np.arange(1995, 2014, 2).tolist()

plt.xticks(years)

labels1 = ('Health expenditure, private (% of GDP), Canada', 'Health expenditure, private (% of GDP), United States',

          'Health expenditure, public (% of GDP), Canada', 'Health expenditure, public (% of GDP), United States')

plt.legend(labels = labels1)

plt.ylim(2, 10)

#plt.legend('Canda', 'US')

plt.savefig('PublicvPrivate Health ExpendituresCanUS.png')

plt.show()
# Extract GDP per Capita of different countries to compare Canada and US GDP per Capita

GDPpC_filter = data['IndicatorName'].str.contains('GDP per capita \(current US', case = False)

df_GDP_per_Capita = data[GDPpC_filter]

df_GDP_per_Capita.head()
# Extract GDP per Capita for Canada and United States only

df_3 = df_GDP_per_Capita.loc[(df_GDP_per_Capita['CountryName'].isin(CanUS_filter))]

df_3.head()
# Create a pivot table for Canada and US GDP per Capita to make plotting easier:

df_4 = pd.pivot_table(df_3, values = ['Value'], index = ['CountryName', 'Year'], columns = ['IndicatorName'])

df_4.head()
# Plot Canada and US GDP per Capita to find out if that is the cause of different health expenditures as percentage of GDP

fig2 = df_4.unstack(level=0).plot(kind = 'line', figsize = (12, 8))

plt.title('GDP per Capita in Canada and the United States', fontdict = fontdict)

plt.ylabel('GDP per Capita, USD', size = 'x-large')

plt.show()
# Limit GDP per Capita comparison to 1995 - 2013 range to represent same years of Health Expenditures

fig3 = df_4.unstack(level=0).plot(kind = 'line', figsize = (12, 8))

plt.ylabel('GDP per Capita, USD', size = 'xx-large')

plt.xlabel('Years', size = 'xx-large')

plt.title('GDP per Capita in Canada and the United States', fontdict = fontdict)

labels2 = ('Canada', 'United States')

plt.legend(labels = labels2, fontsize = 'xx-large')

plt.xticks(years)

plt.xlim(1995, 2013)

plt.savefig('GDPperCapitaCanUS.png')

plt.show()
# Store Canada GDP per Capita from 1995 - 2013 in an array for easier plotting

#gdp_can = df_4.xs('Canada').loc[1995:2013].values.flatten()

#gdp_can
# Find out if there is a correlation between public and private health expenditure (Canada)

import seaborn as sns

h_public_can = df_2.loc['Canada', ('Value', 'Health expenditure, public (% of GDP)')].values

h_private_can = df_2.loc['Canada', ('Value', 'Health expenditure, private (% of GDP)')].values

plt.figure(figsize = (12, 8))

fig4 = sns.regplot(h_public_can, h_private_can)

fig4.set(xlabel = 'Public Health Expenditure (% of GDP)', ylabel = 'Private Health Expenditure (% of GDP)',

         title = 'Public vs. Private Health Expenditure in Canada')

sns.set_style('whitegrid')

sns.set_context('poster')

plt.ylim(2.5, 3.3)

plt.xlim(6, 8)

print(np.corrcoef(h_public_can, h_private_can))

plt.savefig('PublicvPrivate Health Expenditure in Canada.png')

plt.show()
# Is there a correlation between public health expenditure and GDP per Capita

# sns.regplot(h_public_can, gdp_can)
# Find out if there is a correlation between public and private health expenditure (United States)

h_public_us = df_2.loc['United States', ('Value', 'Health expenditure, public (% of GDP)')].values

h_private_us = df_2.loc['United States', ('Value', 'Health expenditure, private (% of GDP)')].values

plt.figure(figsize = (12, 8))

fig5 = sns.regplot(h_public_us, h_private_us)

fig5.set(xlabel = 'Public Health Expenditure (% of GDP)', ylabel = 'Private Health Expenditure (% of GDP)',

         title = 'Public vs. Private Health Expenditure in United States')

plt.ylim(7, 9.5)

plt.xlim(5.5, 8.5)

plt.savefig('PublicvPrivate Health Expenditure in US')

np.corrcoef(h_public_us, h_private_us)
Ind2_filter = ['Hospital beds (per 1,000 people)']

df_5 = df.loc[(df['CountryName'].isin(CanUS_filter)) & (df['IndicatorName'].isin(Ind2_filter))]

df_5.head()
# Pivot Table to make it more convenient for plotting purposes

df_6 = pd.pivot_table(df_5, values = ['Value'], index = ['CountryName', 'Year'], columns = ['IndicatorName'])

df_6.head()
# There are so many NAN values, so drop NAN values

df_6.dropna(inplace = True)

df_6.head()
# plot hospital beds per 1000 people and physicians per 1000 people

df_6.unstack(level=0).plot(style = 'o', figsize = (8, 6))

plt.show()
fig6 = df_6.unstack(level=0).plot(style = 'o', figsize = (12, 8))

fig6.set_xlim(1995, 2013)

plt.xticks(years)

plt.xlabel('Years', fontsize = 'x-large')

plt.ylabel('Hospital beds (per 1000 people)', fontsize = 'x-large')

plt.legend(labels = ('Canada', 'United States'), fontsize = 'x-large')

plt.ylim(2, 6)

plt.savefig('Hospital beds per 1000 people.png')

plt.show()
Ind3_filter = ['Physicians (per 1,000 people)']

df_7 = df.loc[(df['CountryName'].isin(CanUS_filter)) & (df['IndicatorName'].isin(Ind3_filter))]

df_8 = pd.pivot_table(df_7, values = ['Value'], index = ['CountryName', 'Year'], columns = ['IndicatorName'])

fig7 = df_8.unstack(level=0).plot(style = 'o', figsize = (12, 8))

plt.xlim(1995, 2013)

plt.xticks(years)

plt.ylim(1, 3)

plt.yticks([1, 1.5, 2, 2.5, 3])

sns.set_context('notebook')

fig7.set_ylim(1, 3)

plt.xlabel('Years', fontsize = 'x-large')

plt.ylabel('Physicians (per 1000 people)', fontsize = 'x-large')

plt.legend(labels = ('Canada', 'United States'), fontsize = 'x-large')

plt.savefig('Physicians per 1000 people.png')

plt.show()
#fig8, subfigs = plt.subplots(1, 2, figsize = (10, 10))

#subfigs[0].plot()

#subfigs[1] = fig7

#plt.show()
# There is no recovery related indicators, so, crude death rate will be used as an indicator

Ind4_filter = ['Death rate, crude (per 1,000 people)', 'Life expectancy at birth, total (years)']

df_9 = df.loc[(df['CountryName'].isin(CanUS_filter)) & (df['IndicatorName'].isin(Ind4_filter))]

df_10 = pd.pivot_table(df_9, values = ['Value'], index = ['CountryName', 'Year'], columns = ['IndicatorName'])

fig8 = df_10.loc[:, ('Value', 'Death rate, crude (per 1,000 people)')].unstack(level=0).plot(figsize = (10, 7))

fig8.set_xlim(1995, 2013)

fig8.set_ylim(5, 10)

plt.ylabel('Death rate, crude (per 1,000 people)', fontsize = 'x-large')

plt.legend(labels = ('Canada', 'United States'), fontsize = 'x-large')

plt.xlim(1995, 2013)

plt.xticks(years)

plt.savefig('Death rate per 1000 people CanUS.png')

plt.show()
fig9 = df_10.loc[:, ('Value', 'Life expectancy at birth, total (years)')].unstack(level=0).plot(figsize = (10, 7))

fig9.set_xlim(1995, 2013)

plt.ylabel('Life expectancy at birth, total (years)', fontsize = 'x-large')

plt.xticks(years)

plt.legend(labels = ('Canada', 'United States'), fontsize = 'x-large')

plt.xlim(1995, 2013)

plt.ylim(75, 82)

plt.savefig('Life expectancy at birth CanUS.png')

plt.show()
Europe_filter = countries['ShortName'][countries.Region == 'Europe & Central Asia']

#Europe_filter
#Filter countries by high income, OECD

HighIncome_filter = countries['ShortName'][countries.IncomeGroup == 'High income: OECD']

#HighIncome_filter.drop([105], axis = 0, inplace = True)

#HighIncome_filter
# Create a dataframe that contains public and private health expenditures, life expectancy at birth,

#  and death rate for these high income countries
Ind5_filter = ['Death rate, crude (per 1,000 people)', 'Life expectancy at birth, total (years)', 'Health expenditure, public (% of GDP)', 'Health expenditure, private (% of GDP)']

df_11 = df.loc[(df['CountryName'].isin(HighIncome_filter)) & (df['IndicatorName'].isin(Ind5_filter))]

#df_11 = df.loc[(df['IndicatorName'].isin(Ind5_filter))]

df_12 = pd.pivot_table(df_11, values = ['Value'], index = ['CountryName', 'Year'], columns = ['IndicatorName'])

df_12.head()
# No health expenditure data ecist before 1995, so, it is better to remove all data before 1995, dropna is a convenient method

df_12.dropna(axis = 0, inplace = True)

df_12.head()
# Create a new column for total health expenditure

df_12.loc[:, ('Value', 'Total Health Expenditure(% of GDP)')] = df_12.loc[:, ('Value', 'Health expenditure, public (% of GDP)')] + df_12.loc[:, ('Value', 'Health expenditure, private (% of GDP)')]

df_12.head()
# Find if there is a correlation between different indicators

df_12.corr()

df_12.corr().to_csv('High Income OECD Correlations Table.csv')
public_health_oecd = df_12.loc[:, ('Value', 'Health expenditure, public (% of GDP)')].values

life_exp_oecd = df_12.loc[:, ('Value', 'Life expectancy at birth, total (years)')].values

plt.figure(figsize = (12, 8))

fig10 = sns.regplot(public_health_oecd, life_exp_oecd)

#plt.ylim(65, 85)

#plt.yticks(np.arange(65, 5, 85))

plt.ylabel('Life expectancy at birth, (years)', fontsize = 'x-large')

plt.xlabel('Health expenditure, public (% of GDP)', fontsize = 'x-large')

plt.ylim(67, 85)

plt.yticks(np.arange(67, 86, 2).tolist())

plt.xticks(np.arange(1, 12, 2).tolist())

plt.title('Public Health Expenditure (% of GDP) vs. Life Expectancy at Birth (years) - High Income OECD', fontdict = fontdict)

plt.savefig('Public Health Expenditure v Life Expectancy')

plt.show()
LowIncome_filter = countries['ShortName'][countries.IncomeGroup == 'Low income']

df_13 = df.loc[(df['CountryName'].isin(LowIncome_filter)) & (df['IndicatorName'].isin(Ind5_filter))]

df_14 = pd.pivot_table(df_13, values = ['Value'], index = ['CountryName', 'Year'], columns = ['IndicatorName'])

df_14.head()
df_14.dropna(axis = 0, inplace = True)

df_14.loc[:, ('Value', 'Total Health Expenditure(% of GDP)')] = df_14.loc[:, ('Value', 'Health expenditure, public (% of GDP)')] + df_14.loc[:, ('Value', 'Health expenditure, private (% of GDP)')]

df_14.corr().to_csv('Low Income OECD Correlations Table.csv')

df_14.corr()
public_health_lowincome = df_14.loc[:, ('Value', 'Health expenditure, public (% of GDP)')].values

life_exp_lowincome = df_14.loc[:, ('Value', 'Life expectancy at birth, total (years)')].values

plt.figure(figsize = (12, 8))

fig11 = sns.regplot(public_health_lowincome, life_exp_lowincome)

#plt.ylim(65, 85)

#plt.yticks(np.arange(65, 5, 85))

plt.ylabel('Life expectancy at birth, (years)', fontsize = 'x-large')

plt.xlabel('Health expenditure, public (% of GDP)', fontsize = 'x-large')

plt.ylim(30, 70)

plt.yticks(np.arange(30, 71, 5).tolist())

plt.xticks(np.arange(0, 9, 2).tolist())

plt.title('Public Health Expenditure (% of GDP) vs. Life Expectancy at Birth (years) - Low Income', fontdict = fontdict)

plt.savefig('Public Health Expenditure v Life Expectancy - Low Income')

plt.show()