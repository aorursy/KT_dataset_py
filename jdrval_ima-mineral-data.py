# Visualizations on notebook

%matplotlib inline



# Python imports

import os, math, time, random



# Data manipulation

import numpy as np # Linear algebra

import pandas as pd # Data processing





# Data visualization

import matplotlib.pyplot as plt

import geopandas as gpd # Geospatial visualization

import missingno # Missing data visualization

import seaborn as sns # Data visualization

sns.set()



# Text manipulation

import chardet

import fuzzywuzzy

import regex

from pandas_profiling import ProfileReport # Report generator



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
minerals = pd.read_csv('/kaggle/input/ima-database-of-mineral-properties/RRUFF_Export_20191025_022204.csv')



minerals.head()
minerals.tail()
minerals.shape
minerals.info()
def unique_val(column, data):

    unique_values = data[column].unique()

    return unique_values



unique_val('Country of Type Locality', minerals)
missingno.matrix(minerals, figsize=(35,15))
minerals_values_count = minerals.isnull().sum()



minerals_values_count
dupli = minerals.duplicated()

minerals[dupli].shape
minerals.rename(columns={'Mineral Name': 'Mineral', 'RRUFF Chemistry (plain)': 'RRUFF Chem', 'IMA Chemistry (plain)': 'IMA Chem',

                         'Chemistry Elements': 'Elements', 'IMA Number': 'IMA ID', 'RRUFF IDs': 'RRUFF ID', 

                         'Country of Type Locality': 'Country', 'Oldest Known Age (Ma)': 'Age (Ma)'}, inplace=True)

print('Done')
minerals['Country'] = minerals['Country'].fillna(value='Unknown') #  Changing the NaN values to unknown



minerals['Country'].isnull().sum()
# Edited from Daria Chemkaeva, (2020).



countries_splited = minerals.set_index('Mineral')['Country'].str.split(' / ', expand=True).stack().reset_index('Mineral').reset_index(drop=True)

countries_splited.columns = ['Mineral', 'Country']



new_met = countries_splited['Country'].str.contains('meteorite', case=False), 'Country'

new_cou = countries_splited['Country'].str.contains('IDP', case=False), 'Country'



countries_splited.loc[new_met] = 'Meteorite'

countries_splited.loc[new_cou] = 'IDP'

countries_splited['Country'] = countries_splited['Country'].replace({' \?':''}, regex=True)

countries_splited['Country'] = countries_splited['Country'].replace('?', 'Unknown')



print(countries_splited[countries_splited['Country'].str.contains('Unknown', regex=False)])



countries_splited
missing_values_count = countries_splited[countries_splited['Country'].str.contains('Unknown', regex=False)].value_counts()

total_cells = np.product(minerals['Country'].shape)

total_missing = missing_values_count.sum()

percent_missing_country = (total_missing/total_cells) * 100

print(percent_missing_country)
countries_splited['Country'].value_counts()
countries_splited.drop(countries_splited[countries_splited['Country'] == 'Unknown'].index, inplace=True)



countries_splited
null_elements = minerals['Elements'].isnull()

minerals[null_elements]
missing_values_count = minerals['Elements'].isnull().sum()

total_cells = np.product(minerals['Elements'].shape)

total_missing = missing_values_count.sum()

percent_missing_elems = (total_missing/total_cells) * 100

print(percent_missing_elems)
del_rows = minerals['Elements'].notna()

minerals = minerals[del_rows] #  New DF without the 4 NaN rows



minerals['Elements'].isna().sum()
# Edited from Daria Chemkaeva, (2020).



elements_splited = minerals.set_index('Mineral')['Elements'].str.split(' ', expand=True).stack().reset_index('Mineral').reset_index(drop=True)

elements_splited.columns = ['Mineral', 'Element']



elements_splited
minerals['Age (Ma)'].unique()
minerals['Age (Ma)'] = minerals['Age (Ma)'].astype('float64')

print('Done')
minerals['Age (Ma)'] = minerals['Age (Ma)'].replace(0, 'Unknown')



minerals['Age (Ma)']
minerals['Age (Ma)'] = minerals['Age (Ma)'].fillna(value='Unknown') #  Changing the NaN values to unknown



minerals['Age (Ma)']
minerals['Age (Ma)'].isna().sum()



missing_values_count = minerals['Age (Ma)'].str.contains('Unknown', regex=False).value_counts()

total_cells = np.product(minerals['Age (Ma)'].shape)

total_missing = missing_values_count.sum()

percent_missing_age = (total_missing/total_cells) * 100

print('Number of missing cells: ', missing_values_count[:1])

print('% = ', percent_missing_age)
minerals['Age (Ma)'].value_counts()[0:25]
age_removed = minerals[minerals['Age (Ma)'] != 'Unknown']

                                         

age_removed['Age (Ma)'].value_counts().sort_index(ascending=False)
oldest_mine = age_removed.groupby('Mineral')[['Country', 'Age (Ma)']].max()



oldest_mine = oldest_mine.sort_values(by='Age (Ma)', ascending=False)[0:10]



oldest_min_notna = oldest_mine[~oldest_mine['Country'].str.contains('unknown')]



oldest_min_notna
plt.figure(figsize=(40,10))

plt.title("Quantity of minerals per country")

sns.barplot(x=countries_splited['Country'].value_counts()[0:10].index, y=countries_splited['Country'].value_counts()[0:10])

plt.ylabel("Amount of minerals")

plt.xlabel("Countries", labelpad=14)
countries_splited['Country'].value_counts()[0:10]
countries_splited['Country'].value_counts()
elements_splited['Element'].value_counts()
filtro_1 = countries_splited['Country'].map(countries_splited['Country'].value_counts()) >= 200

elem_1 = countries_splited[filtro_1]



filtro_2 = elements_splited['Element'].map(elements_splited['Element'].value_counts()) >= 500

elem_2 = elements_splited[filtro_2]



plot_country_elem = pd.merge(elem_1, elem_2, on='Mineral')



plot_country_elem
plt.figure(figsize=(30,5))

plt.title("Most abundant chemistry element")

sns.lineplot(data=plot_country_elem.Element.value_counts())

plt.ylabel("Quantity")

plot_country_elem.Element.value_counts()
# Edited from Daria Chemkaeva, (2020).



sns.catplot(x="Country", hue="Element", kind="count", palette="colorblind", edgecolor=".01", data=plot_country_elem, height=9, aspect=2)
oldest_min = age_removed.groupby('Mineral')[['Country', 'Age (Ma)']].max()



oldest_min = oldest_min.sort_values(by='Age (Ma)', ascending=False)



oldest_min_notna = oldest_min[~oldest_min['Country'].str.contains('unknown')]



oldest_min_notna[0:10]
plt.figure(figsize=(35,6))

plt.title("Oldests minerals")

sns.barplot(x=oldest_min_notna[0:10].index, y=oldest_min_notna['Age (Ma)'][0:10])

plt.ylabel("Quantity")
plt.figure(figsize=(35,5))

plt.title("Countries with the oldest minerals")

sns.barplot(x=oldest_min_notna['Country'][0:10], y=oldest_min_notna['Age (Ma)'])

plt.ylabel("Age (Ma)")
age_group = oldest_min_notna.groupby('Mineral')[['Country', 'Age (Ma)']].max()

x = age_group.groupby(['Age (Ma)', 'Country'])['Age (Ma)'].unique()        #.apply(lambda df: df['Age (Ma)']).sort_values()

x.sort_values(ascending=False)

sns.set(font_scale=2)

sns.set_style("white")

plt.figure(figsize=(35,10))

plt.title("Quantity of minerals per country")

sns.barplot(x=countries_splited['Country'].value_counts()[0:10].index, y=countries_splited['Country'].value_counts()[0:10])

plt.ylabel("Amount of minerals")

plt.xlabel("Countries", labelpad=14)

countries_splited['Country'].value_counts()[0:10]
filtro_1 = countries_splited['Country'].map(countries_splited['Country'].value_counts()) >= 200

elem_1 = countries_splited[filtro_1]



filtro_2 = elements_splited['Element'].map(elements_splited['Element'].value_counts()) >= 500

elem_2 = elements_splited[filtro_2]



sns.set(font_scale=2)

sns.set_style("white")

plot_country_elem = pd.merge(elem_1, elem_2, on='Mineral')



sns.catplot(x="Country", hue="Element", kind="count", palette="colorblind", edgecolor=".01", data=plot_country_elem, height=9, aspect=3)
sns.set(font_scale=2)

sns.set_style("white")

plt.figure(figsize=(35,10))

plt.title("Most abundant chemistry element")

sns.lineplot(data=plot_country_elem.Element.value_counts())

plt.ylabel("Quantity")



plot_country_elem.Element.value_counts()
sns.set(font_scale=2)

sns.set_style("white")

plt.figure(figsize=(35,10))

plt.title("Oldests minerals")

sns.barplot(x=oldest_min_notna[0:10].index, y=oldest_min_notna['Age (Ma)'][0:10])

plt.ylabel("Quantity")

sns.set(font_scale=2)

plt.figure(figsize=(35,15))

plt.title("Countries with the oldest minerals")

sns.barplot(x=oldest_min_notna['Country'][0:5], y=oldest_min_notna['Age (Ma)'])

plt.ylabel("Age (Ma)")
sns.set(font_scale=2)

plt.figure(figsize=(35,15))

plt.title("Most abundant age")

sns.lineplot(data=oldest_min_notna['Age (Ma)'].value_counts())

plt.ylabel("Quantity")



oldest_min_notna['Age (Ma)'].value_counts(ascending=False)
sns.set(font_scale=2)

sns.set_style("white")

plt.figure(figsize=(35,15))

sns.distplot(a=oldest_min_notna['Age (Ma)'], kde=True)

plt.title("Histogram of Age (Ma)")

plt.legend()
sns.set_style(style="white")

rs = np.random.RandomState(5000)

new_data = pd.DataFrame(data=rs.normal(size=(200, 14)), columns=list(minerals.columns))

corr = new_data.corr()

matrix = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(7, 10))

cmap = sns.diverging_palette(10, 190, as_cmap=True)

sns.heatmap(corr, mask=matrix, cmap=cmap, vmax=.2, center=0, square=True, cbar_kws={"shrink": .55})



rep = ProfileReport(minerals)

rep.to_file(output_file='report.html')
rep