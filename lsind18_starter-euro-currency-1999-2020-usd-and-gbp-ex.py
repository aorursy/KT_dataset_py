from IPython.display import HTML



HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/dIUktr3Zpyk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')
df_cur = pd.read_csv("/kaggle/input/euro-exchange-daily-rates-19992020/euro-daily-hist_1999_2020.csv", parse_dates=["Period\\Unit:"])

df_cur.sample(5)
names = str.maketrans('', '', '[]')

df_cur.columns = df_cur.columns.str.translate(names)

df_cur.columns = df_cur.columns.str.strip()

df_cur.set_index('Period\\Unit:', inplace=True)

df_cur.index.rename('DateSeries', inplace = True)

df_cur.info()
cols = list(df_cur)

df_cur[cols] = df_cur[cols].apply(pd.to_numeric, errors='coerce')

df_cur.info()
df_cur.isnull().sum(axis = 0)
n = df_cur.index[df_cur.isnull().all(1)]

print(n)

print('Number of NaN rows: {}'.format(len(n)))
df_cur = df_cur.drop(n)
#df_cur = df_cur.fillna(method='backfill')
df_cur.describe(include='all')
df_cur1 = df_cur.reset_index()

df_melted=df_cur1.melt(id_vars=['DateSeries'], var_name='Currency name', value_name='Value')

df_melted.head(5)
dataUSDGBP = df_melted.loc[(df_melted['Currency name'] == 'US dollar') | (df_melted['Currency name'] == 'UK pound sterling')]

dataUSDGBP.sample(5)
fig = plt.figure(figsize=(15,8))

plt.grid(which='major', linewidth = 2)

plt.minorticks_on()

plt.grid(which='minor', linewidth = 0.5)

sns.lineplot(x='DateSeries', y='Value', hue='Currency name', data = dataUSDGBP)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
dataUSD = dataUSDGBP.loc[(dataUSDGBP['Currency name'] == 'US dollar')]

dataUSD.set_index('DateSeries', inplace=True)

print('------USD: 5 largest values by dates------')

print(dataUSD['Value'].nlargest().sort_values(ascending = False))

print('------USD: 5 smallest values by dates-----')

print(dataUSD['Value'].nsmallest().sort_values(ascending = False))
dataGBP = dataUSDGBP.loc[(dataUSDGBP['Currency name'] == 'UK pound sterling')]

dataGBP.set_index('DateSeries', inplace=True)

print('------GBP: 5 largest values by dates------')

print(dataGBP['Value'].nlargest().sort_values(ascending = False))

print('------GBP: 5 smallest values by dates-----')

print(dataGBP['Value'].nsmallest().sort_values(ascending = False))
dataSIT = df_melted.loc[(df_melted['Currency name'] == 'Slovenian tolar')]

dataSIT;
fig = plt.figure(figsize=(15,8))



plt.grid(which='major', linewidth = 2)

plt.minorticks_on()

plt.grid(which='minor', linewidth = 0.5)

sns.lineplot(x='DateSeries', y='Value', hue='Currency name', data = dataSIT)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);