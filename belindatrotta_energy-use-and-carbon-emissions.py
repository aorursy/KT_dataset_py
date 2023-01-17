import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read data
country = pd.read_csv('../input/Country.csv')
country_notes = pd.read_csv('../input/CountryNotes.csv')
indicators = pd.read_csv('../input/Indicators.csv')
series = pd.read_csv('../input/Series.csv')
series_notes = pd.read_csv('../input/SeriesNotes.csv')

# extract relevant indicators
column_list = ['Population, total', 'CO2 emissions (kt)', 
               'CO2 intensity (kg per kg of oil equivalent energy use)',
               'Alternative and nuclear energy (% of total energy use)',
               'Energy use (kg of oil equivalent per capita)']
carbon_indicators = indicators[indicators['IndicatorName'].isin(column_list)]

# separate "countries" that are really aggregates
aggregates = country[np.logical_and(country['SpecialNotes'].str.len() > 0, country['SpecialNotes'].str.contains('aggregate', case=False))]['TableName'].values
countries = country[~country['TableName'].isin(aggregates)]['TableName'].values

# reshape dataframe
carbon_indicators.drop(['IndicatorCode','CountryCode'],axis=1,inplace=True)
carbon_indicators.set_index(['CountryName','Year','IndicatorName'],inplace=True)
carbon_indicators = carbon_indicators.unstack('IndicatorName')
carbon_indicators.columns = carbon_indicators.columns.droplevel(0)

# give columns shorter names
carbon_indicators = carbon_indicators[column_list]  # re-order the columns, since we don't know what order they're in
carbon_indicators.columns = ['Population', 'Carbon emissions', 'Carbon intensity', 'Alternative energy percent','Energy use per capita']

# calculate per-capita emissions
carbon_indicators['Emissions per capita'] = carbon_indicators['Carbon emissions'] / carbon_indicators['Population']

# calculate total energy use
carbon_indicators['Energy use'] = carbon_indicators['Energy use per capita'] * carbon_indicators['Population']
plt.figure()
plt.plot(carbon_indicators.index.levels[1], carbon_indicators.loc[pd.IndexSlice['World',:],'Carbon emissions'])
plt.title('Carbon emissions, kt')

plt.figure()
plt.plot(carbon_indicators.index.levels[1], carbon_indicators.loc[pd.IndexSlice['World',:],'Population'])
plt.title('Population')

plt.figure()
plt.plot(carbon_indicators.index.levels[1], carbon_indicators.loc[pd.IndexSlice['World',:],'Emissions per capita'])
plt.title('Emissions per capita, kt')
plt.figure()
plt.plot(carbon_indicators.index.levels[1], carbon_indicators.loc[pd.IndexSlice['World',:],'Alternative energy percent'])
plt.title('Alternative energy percent')


top_energy_users_2000 = carbon_indicators.loc[pd.IndexSlice[list(countries),2000],['Energy use', 'Alternative energy percent']].sort_values('Energy use', ascending=False).iloc[0:5]
proportion_of_total = sum(top_energy_users_2000['Energy use']) / carbon_indicators.loc[pd.IndexSlice['World',2000],'Energy use']
print('Top 5 energy users in 2000 (these account for {:.0f}% of total energy use)'.format(proportion_of_total*100))
print(top_energy_users_2000)

top_energy_users_2011 = carbon_indicators.loc[pd.IndexSlice[list(countries),2011],['Energy use', 'Alternative energy percent']].sort_values('Energy use', ascending=False).iloc[0:5]
proportion_of_total = sum(top_energy_users_2011['Energy use']) / carbon_indicators.loc[pd.IndexSlice['World',2011],'Energy use']
print('\n\nTop 5 energy users in 2000 (these account for {:.0f}% of total energy use)'.format(proportion_of_total*100))
print(top_energy_users_2011)