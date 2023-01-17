import pandas as pd      # data processing, CSV file I/O

import seaborn as sns    # beautiful plots

sns.set()

import matplotlib.pyplot as plt

%matplotlib inline
minerals = pd.read_csv("/kaggle/input/ima-database-of-mineral-properties/RRUFF_Export_20191025_022204.csv")

minerals.sample(5)
minerals.rename(columns={'Mineral Name':'Mineral','Chemistry Elements':'Elements', 'Country of Type Locality' : 'Country', 'Crystal Systems':'Systems', 'Oldest Known Age (Ma)' : 'Age', 'Structural Groupname' : 'Groupname' }, inplace=True)
minerals.isnull().sum(axis = 0)
el_null_mask = minerals['Elements'].isnull() | minerals['IMA Chemistry (plain)'].isnull()

minerals[el_null_mask]
minerals = minerals[minerals.Elements.notna()]
minerals['Country'] = minerals.Country.fillna(value = 'unknown')

minerals.isnull().sum(axis = 0)
elem = minerals.set_index('Mineral').Elements.str.split(' ', expand=True).stack().reset_index('Mineral').reset_index(drop=True)

elem.columns = ['Mineral', 'Element']

elem
elem['Element'].value_counts()
elem['Element'].value_counts()[0:10].sort_values().plot(kind='barh', figsize=(8, 6))

plt.xlabel("Count of occurrences", labelpad=14)

plt.ylabel("Chemistry Element", labelpad=14)

plt.title("Most frequent elements in minerals", y=1.02);
minerals['Country'].value_counts()
countr = minerals.set_index('Mineral').Country.str.split(' / ', expand=True).stack().reset_index('Mineral').reset_index(drop=True)

countr.columns = ['Mineral', 'Country']
print(countr[countr['Country'].str.contains('meteorite', regex=False)])
print(countr[countr['Country'].str.contains('?', regex=False)])
countr['Country'] = countr['Country'].replace({' \?':''}, regex=True)

countr.loc[countr['Country'].str.contains('meteorite', case=False), 'Country'] = 'meteorite'

countr['Country'] = countr['Country'].replace('?', 'unknown')

countr.loc[countr['Country'].str.contains('IDP', case=False), 'Country'] = 'IDP'
countr['Country'].str.strip()

countr.drop(countr[countr.Country == 'unknown'].index, inplace=True)

countr['Country'].value_counts()[0:20]
countr['Country'].value_counts()[0:20].sort_values().plot(kind='barh', figsize=(10, 7))

plt.xlabel("Count of occurancies", labelpad=14)

plt.ylabel("Country", labelpad=14)

plt.title("Count countries by minerals", y=1.02);
countr = countr[countr['Country'].map(countr['Country'].value_counts()) >= 140]

elem = elem[elem['Element'].map(elem['Element'].value_counts()) >= 600]
result = pd.merge(countr, elem, on='Mineral')

result
sns.catplot(x="Country", hue="Element", kind="count", palette="pastel", edgecolor=".6", data=result, height=8, aspect = 2)