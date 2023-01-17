import pandas as pd
import_data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')
import_data.head()
import_data.shape
unique_commodities = import_data['Commodity'].unique()
len(unique_commodities)
unique_countries = import_data['country'].unique()
len(unique_countries)
unique_year = import_data['year'].unique()
len(unique_year)
yearly_data = import_data.groupby('year')['value'].sum()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
font_dict = {

    'size': 20,

    'weight': 'bold'

}
plt.figure(figsize=(10, 4))

plt.subplot(211)

plt.title("Import Quantity-vs-Year", fontdict=font_dict)

plt.bar(x=yearly_data.index, height=yearly_data.values)

plt.ylabel('Import Quantity')

plt.subplot(212)

plt.plot(yearly_data)

plt.xlabel('Year')

plt.ylabel('Import Quantity')

plt.show()
country_import = import_data.groupby(['country', 'year'])['value'].sum()
country_import = country_import.sort_index(axis=0, ascending=True)
country_import
country_import.loc['AFGHANISTAN TIS']
plt.figure(figsize=(10, 8))

plt.pie(country_import.loc['AFGHANISTAN TIS'].values, labels=country_import.loc['AFGHANISTAN TIS'].index,

       autopct="%.2f")

plt.title("Trade Analysis ", fontdict=font_dict)
import_data['Commodity'].unique()[:20]
commodities_2016_sum = import_data[import_data['year'] == 2016].groupby('Commodity')['value'].sum()
commodities_2016_sum.sort_values(ascending=False).head(1).index
commodities_2018_sum = import_data[import_data['year'] == 2018].groupby('Commodity')['value'].sum()
commodities_2018_sum.sort_values(ascending=False).head(1).index
commodities_sum = import_data.groupby('Commodity')['value'].sum().sort_values(ascending=False)
commodities_sum = commodities_sum[:10]
## reduce the name of the commodities so that it's easier to visualize

commodities_sum.index = commodities_sum.index.map(lambda x: x[:15])
plt.figure(figsize=(10, 8))

plt.pie(commodities_sum, labels=commodities_sum.index, autopct="%.2f")

plt.title('Resource-vs-Import Share Percent', fontdict=font_dict)
import_data['country'][import_data['country'].str.startswith('U')].unique()
import_data['country'][import_data['country'].str.startswith('C')].unique()
import_data['country'][import_data['country'].str.startswith('I')].unique()
import_data['country'][import_data['country'].str.startswith('J')].unique()
import_data['country'][import_data['country'].str.startswith('R')].unique()
import_data['country'][import_data['country'].str.startswith('A')].unique()
import_data['country'][import_data['country'].str.startswith('P')].unique()
countries_list = ['U S A', 'U ARAB EMTS', 'U K', 'CHINA P RP', 'IRAN', 'JAPAN', 'RUSSIA', 'AFGHANISTAN TIS', 'PAKISTAN IR']
specific_countries_data = import_data[import_data['country'].isin(countries_list)].groupby(['country', 'year'])['value'].sum()
specific_countries_data
countries_share = import_data[import_data['country'].isin(countries_list)].groupby('country')['value'].sum().sort_values(ascending=False)
countries_share = countries_share.apply(lambda x: round(100 * x/countries_share.sum()))
countries_share
plt.figure(figsize=(12, 8))

plt.pie(countries_share, labels=countries_share.index, autopct="%.2f")

plt.title('Countries Percentage Share', fontdict=font_dict)
plt.figure(figsize=(10, 8))

plt.subplot(211)

plt.bar(x=specific_countries_data.loc['CHINA P RP'].index, height=specific_countries_data.loc['CHINA P RP'].values)

plt.title('Import-vs-Year for China', fontdict=font_dict)

plt.subplot(212)

plt.plot(specific_countries_data.loc['CHINA P RP'])

plt.xlabel('Year')

plt.ylabel('Import in US($)')
for country in countries_list:

    plt.figure(figsize=(10, 8))

    plt.subplot(211)

    plt.bar(x=specific_countries_data.loc[country].index, height=specific_countries_data.loc[country].values)

    title_name = 'Import-vs-Year for ' + country

    plt.title(title_name, fontdict=font_dict)

    plt.subplot(212)

    plt.plot(specific_countries_data.loc[country])

    plt.xlabel('Year')

    plt.ylabel('Import in US($)')