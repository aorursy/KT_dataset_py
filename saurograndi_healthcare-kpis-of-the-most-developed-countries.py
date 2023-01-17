# Data Source: https://www.kaggle.com/worldbank/world-development-indicators
# Folder: 'world-development-indicators'
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
data = pd.read_csv('../input/Indicators.csv')
data.shape
# How many unique country codes and indicator names are there ? 
countries = data['CountryCode'].unique().tolist()
indicators = data['IndicatorName'].unique().tolist()
print('How many unique country codes and indicator names are there?')
print('Contries size: %d' % len(countries))
print('Indicators size: %d' % len(indicators))
mask = data['IndicatorName'].str.contains('Health|Hospital|Physician|Population ages')
indicatorsFilter = data[mask]['IndicatorName'].unique().tolist()
#indicatorsFilter
indicatorsFilter = ['Health expenditure per capita (current US$)',
                    'Hospital beds (per 1,000 people)', 
                    'Physicians (per 1,000 people)'
                    ]
countriesFilter = ['CAN', 'FRA', 'DEU', 'IT', 'JPN', 'GBR', 'USA', 
                  'CHE', 'DNK', 'NLD', 'BEL', 'SWE', 'NOR', 'FIN']
yearsFilter = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
print ('No data available for the following Countries-Indicator:')
for country in countriesFilter:
    for indicator in indicatorsFilter:
        df = data[(data['CountryCode'] == country) &  (data['IndicatorName'] == indicator)]
        size = df.size
        if size == 0:
            print('Country %s, Indicator %s, size %d'% (country, indicator, size))
# Let's remove China and Italy from the countriesFilter
countriesFilter = ['CAN', 'FRA', 'DEU', 'JPN', 'GBR', 'USA', 
                  'CHE', 'DNK', 'NLD', 'BEL', 'SWE', 'NOR', 'FIN']
# Let's reduce the dataset extracting data by CountryCode and IndicatorName corresponding to our choice
filterFull = (data['CountryCode'].isin(countriesFilter)) & (data['IndicatorName'].isin(indicatorsFilter)) & (data['Year'].isin(yearsFilter))
data = data.loc[filterFull]
health_exp_df = data[data['IndicatorName'] == 'Health expenditure per capita (current US$)']
hosp_bed_df = data[data['IndicatorName'] == 'Hospital beds (per 1,000 people)']
phys_df = data[data['IndicatorName'] == 'Physicians (per 1,000 people)']
# Dataset size for each country and each indicator
# We accept maximum 3 years of missing data
print(health_exp_df['IndicatorName'].iloc[0])
for country in countriesFilter:
    df = health_exp_df[health_exp_df['CountryCode'] == country]
    if df.shape[0] < 8:
        print(country + ': ' + 'has more that 3 years of missing data')

print(hosp_bed_df['IndicatorName'].iloc[0])
for country in countriesFilter:
    df = hosp_bed_df[hosp_bed_df['CountryCode'] == country]
    if df.shape[0] < 8:
        print(country + ': ' + 'has more that 3 years of missing data')
        
print(phys_df['IndicatorName'].iloc[0])    
for country in countriesFilter:
    df = phys_df[phys_df['CountryCode'] == country]
    if df.shape[0] < 8:
        print(country + ': ' + 'has more that 3 years of missing data')
# dataframe with only the Year column
date_df = pd.DataFrame({'Year' : [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,2010]})
hosp_bed_FRA_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'FRA'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_DEU_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'DEU'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_GBR_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'GBR'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_DNK_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'DNK'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_BEL_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'BEL'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_NOR_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'NOR'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_FIN_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'FIN'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_CAN_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'CAN'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_USA_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'USA'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
hosp_bed_JPN_merged = hosp_bed_df[hosp_bed_df['CountryCode'] == 'JPN'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
phys_FRA_merged = phys_df[phys_df['CountryCode'] == 'FRA'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
phys_DEU_merged = phys_df[phys_df['CountryCode'] == 'DEU'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
phys_DNK_merged = phys_df[phys_df['CountryCode'] == 'DNK'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
phys_NLD_merged = phys_df[phys_df['CountryCode'] == 'NLD'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
phys_SWE_merged = phys_df[phys_df['CountryCode'] == 'SWE'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
phys_FIN_merged = phys_df[phys_df['CountryCode'] == 'FIN'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
phys_NOR_merged = phys_df[phys_df['CountryCode'] == 'NOR'].merge(date_df, on='Year', how='outer').sort_values(by='Year', ascending=True).fillna(method='ffill')
# Extract a list ov values per Country
health_exp_FRA = health_exp_df[health_exp_df['CountryCode'] == 'FRA']['Value'].values
health_exp_DEU = health_exp_df[health_exp_df['CountryCode'] == 'DEU']['Value'].values
health_exp_GBR = health_exp_df[health_exp_df['CountryCode'] == 'GBR']['Value'].values
health_exp_CHE = health_exp_df[health_exp_df['CountryCode'] == 'CHE']['Value'].values
health_exp_DNK = health_exp_df[health_exp_df['CountryCode'] == 'DNK']['Value'].values
health_exp_NLD = health_exp_df[health_exp_df['CountryCode'] == 'NLD']['Value'].values
health_exp_BEL = health_exp_df[health_exp_df['CountryCode'] == 'BEL']['Value'].values
health_exp_SWE = health_exp_df[health_exp_df['CountryCode'] == 'SWE']['Value'].values
health_exp_NOR = health_exp_df[health_exp_df['CountryCode'] == 'NOR']['Value'].values
health_exp_FIN = health_exp_df[health_exp_df['CountryCode'] == 'FIN']['Value'].values
health_exp_CAN = health_exp_df[health_exp_df['CountryCode'] == 'CAN']['Value'].values
health_exp_USA = health_exp_df[health_exp_df['CountryCode'] == 'NLD']['Value'].values
health_exp_JPN = health_exp_df[health_exp_df['CountryCode'] == 'JPN']['Value'].values
# Bar Chart
years = np.array(yearsFilter)
width = 0.05

fig, ax = plt.subplots(figsize=(15, 8))

# create
plt_FRA = ax.bar(years,health_exp_FRA, width)
plt_DEU = ax.bar(years + width,health_exp_DEU, width)
plt_GBR = ax.bar(years + 2*width,health_exp_GBR, width)
plt_CHE = ax.bar(years + 3*width,health_exp_CHE, width)
plt_DNK = ax.bar(years + 4*width,health_exp_DNK, width)
plt_NLD = ax.bar(years + 5*width,health_exp_NLD, width)

plt_BEL = ax.bar(years + 6*width,health_exp_BEL, width)
plt_SWE = ax.bar(years + 7*width,health_exp_SWE, width)
plt_NOR = ax.bar(years + 8*width,health_exp_NOR, width)
plt_FIN = ax.bar(years + 9*width,health_exp_FIN, width)
plt_CAN = ax.bar(years + 10*width,health_exp_CAN, width)
plt_USA = ax.bar(years + 11*width,health_exp_USA, width)
plt_JPN = ax.bar(years + 12*width,health_exp_JPN, width)

# Axes and Labels
ax.set_xlim(years[0]-3*width, years[len(years)-1]+10*width)
ax.set_xlabel('Year')
ax.set_xticks(years+2*width)
xtickNames = ax.set_xticklabels(years)
plt.setp(xtickNames, rotation=45, fontsize=10)

ax.set_ylabel(health_exp_df['IndicatorName'].iloc[0])
#label the figure
ax.set_title(health_exp_df['IndicatorName'].iloc[0])
ax.legend( (plt_FRA[0], plt_DEU[0], plt_GBR[0], plt_CHE[0], plt_DNK[0], plt_NLD[0], 
          plt_BEL[0], plt_SWE[0], plt_NOR[0], plt_FIN[0], plt_CAN[0], plt_USA[0], plt_JPN[0]),
          ('FRA', 'DEU', 'GBR', 'CHE', 'DNK', 'NLD', 'BEL', 'SWE', 'NOR', 'FIN', 'CAN', 'USA', 'JPN') )

plt.show()
# Extract a list ov values per Country
hosp_bed_FRA = hosp_bed_FRA_merged[hosp_bed_FRA_merged['CountryCode'] == 'FRA']['Value'].values
hosp_bed_DEU = hosp_bed_DEU_merged[hosp_bed_DEU_merged['CountryCode'] == 'DEU']['Value'].values
hosp_bed_GBR = hosp_bed_GBR_merged[hosp_bed_GBR_merged['CountryCode'] == 'GBR']['Value'].values
hosp_bed_DNK = hosp_bed_DNK_merged[hosp_bed_DNK_merged['CountryCode'] == 'DNK']['Value'].values
hosp_bed_BEL = hosp_bed_BEL_merged[hosp_bed_BEL_merged['CountryCode'] == 'BEL']['Value'].values
hosp_bed_NOR = hosp_bed_NOR_merged[hosp_bed_NOR_merged['CountryCode'] == 'NOR']['Value'].values
hosp_bed_FIN = hosp_bed_FIN_merged[hosp_bed_FIN_merged['CountryCode'] == 'FIN']['Value'].values
hosp_bed_CAN = hosp_bed_CAN_merged[hosp_bed_CAN_merged['CountryCode'] == 'CAN']['Value'].values
hosp_bed_USA = hosp_bed_USA_merged[hosp_bed_USA_merged['CountryCode'] == 'USA']['Value'].values
hosp_bed_JPN = hosp_bed_JPN_merged[hosp_bed_JPN_merged['CountryCode'] == 'JPN']['Value'].values
years = np.array(yearsFilter)
width = 0.05

fig, ax = plt.subplots(figsize=(12, 8))

# create
plt_FRA = ax.bar(years,hosp_bed_FRA, width)
plt_DEU = ax.bar(years + width,hosp_bed_DEU, width)
plt_GBR = ax.bar(years + 2*width,hosp_bed_GBR, width)
plt_DNK = ax.bar(years + 3*width,hosp_bed_DNK, width)
plt_BEL = ax.bar(years + 4*width,hosp_bed_BEL, width)
plt_NOR = ax.bar(years + 5*width,hosp_bed_NOR, width)
plt_FIN = ax.bar(years + 6*width,hosp_bed_FIN, width)
plt_CAN = ax.bar(years + 7*width,hosp_bed_CAN, width)
plt_USA = ax.bar(years + 8*width,hosp_bed_USA, width)
plt_JPN = ax.bar(years + 9*width,hosp_bed_JPN, width)

# Axes and Labels
ax.set_xlim(years[0]-3*width, years[len(years)-1]+10*width)
ax.set_xlabel('Year')
ax.set_xticks(years+2*width)
xtickNames = ax.set_xticklabels(years)
plt.setp(xtickNames, rotation=45, fontsize=10)

ax.set_ylabel(hosp_bed_df['IndicatorName'].iloc[0])
#label the figure
ax.set_title(hosp_bed_df['IndicatorName'].iloc[0])
ax.legend( (plt_FRA[0], plt_DEU[0], plt_GBR[0], plt_DNK[0],
          plt_BEL[0], plt_NOR[0], plt_FIN[0], plt_CAN[0], plt_USA[0], plt_JPN[0]),
          ('FRA', 'DEU', 'GBR', 'DNK', 'BEL', 'NOR', 'FIN', 'CAN', 'USA', 'JPN') )

plt.show()
# Extract a list ov values per Country
phys_FRA = phys_DEU_merged[phys_FRA_merged['CountryCode'] == 'FRA']['Value'].values
phys_DEU = phys_DEU_merged[phys_DEU_merged['CountryCode'] == 'DEU']['Value'].values
phys_DNK = phys_DNK_merged[phys_DNK_merged['CountryCode'] == 'DNK']['Value'].values
phys_NLD = phys_NLD_merged[phys_NLD_merged['CountryCode'] == 'NLD']['Value'].values
phys_SWE = phys_SWE_merged[phys_SWE_merged['CountryCode'] == 'SWE']['Value'].values
phys_FIN = phys_FIN_merged[phys_FIN_merged['CountryCode'] == 'FIN']['Value'].values
phys_NOR = phys_NOR_merged[phys_NOR_merged['CountryCode'] == 'NOR']['Value'].values
years = np.array(yearsFilter)
width = 0.1

fig, ax = plt.subplots(figsize=(15, 5))

# create
plt_FRA = ax.bar(years,hosp_bed_FRA, width)
plt_DEU = ax.bar(years + width,hosp_bed_DEU, width)
plt_DNK = ax.bar(years + 2*width,phys_DNK, width)
plt_NLD = ax.bar(years + 3*width,phys_NLD, width)
plt_SWE = ax.bar(years + 4*width,phys_SWE, width)
plt_FIN = ax.bar(years + 5*width,phys_FIN, width)
plt_NOR = ax.bar(years + 6*width,phys_NOR, width)

# Axes and Labels
ax.set_xlim(years[0]-3*width, years[len(years)-1]+10*width)
ax.set_xlabel('Year')
ax.set_xticks(years+2*width)
xtickNames = ax.set_xticklabels(years)
plt.setp(xtickNames, rotation=45, fontsize=10)

ax.set_ylabel(phys_df['IndicatorName'].iloc[0])
#label the figure
ax.set_title(phys_df['IndicatorName'].iloc[0])
ax.legend( (plt_FRA[0], plt_DEU[0], plt_DNK[0],
          plt_NLD[0], plt_SWE[0], plt_FIN[0], plt_NOR[0]),
          ('FRA', 'DEU', 'DNK', 'NLD', 'SWE', 'FIN', 'NOR') )

plt.show()
# Reset Contry filter to have all data available for both indicators
countriesFilter = ['CAN', 'FRA', 'DEU', 'JPN', 'GBR', 'USA', 
                   'DNK', 'BEL', 'FIN']
filterCorr = (data['CountryCode'].isin(countriesFilter)) & (data['IndicatorName'] == 'Health expenditure per capita (current US$)') & (data['Year'].isin(yearsFilter))

# Extract the two datasets
health_exp_df = data[filterCorr]
hosp_bed_df = pd.concat([hosp_bed_FRA_merged,hosp_bed_CAN_merged,hosp_bed_DEU_merged,
                    hosp_bed_JPN_merged,hosp_bed_GBR_merged,hosp_bed_USA_merged,
                    hosp_bed_DNK_merged,hosp_bed_BEL_merged,hosp_bed_FIN_merged])
%matplotlib inline
import matplotlib.pyplot as plt

fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.set_title('Health expenditure per capita vs. Hospital beds (per 1,000 people)',fontsize=10)
axis.set_xlabel(health_exp_df['IndicatorName'].iloc[0],fontsize=10)
axis.set_ylabel(hosp_bed_df['IndicatorName'].iloc[0],fontsize=10)

X = health_exp_df['Value']
Y = hosp_bed_df['Value']

axis.scatter(X, Y)
plt.show()
np.corrcoef(health_exp_df['Value'],hosp_bed_df['Value'])
