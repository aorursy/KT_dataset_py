import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ekspor = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')

impor = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')
impor.info()
impor = impor.drop_duplicates()

ekspor = ekspor.drop_duplicates()
impor.info()
impor['value'] = -1*impor['value']

neraca = pd.concat([ekspor,impor])

neraca.head()
plot = []

for i in neraca['year'].unique():

    plot.append([i, neraca[neraca['year'] == i]['value'].sum(), 

                 neraca[(neraca['year'] == i) & (neraca['value'] >= 0)]['value'].sum(),

                 neraca[(neraca['year'] == i) & (neraca['value'] <= 0)]['value'].sum()])



sumOfTrade = pd.DataFrame(plot, columns = ['year', 'total','export','import'])

sumOfTrade.head()
plt.figure(figsize = (8,8))

# plt.bar(sumOfTrade['year'],sumOfTrade['total'])

plt.bar(sumOfTrade['year'],sumOfTrade['export'],color = 'g', label = 'export')

plt.bar(sumOfTrade['year'],sumOfTrade['import'],color = 'r', label = 'import')

plt.legend()

plt.title('India\'s export & import sum')

plt.xlabel('Year')

plt.ylabel('Value [USD]')
plot = []

for i in neraca['country'].unique():

    plot.append([i, neraca[neraca['country'] == i]['value'].sum(), 

                 neraca[(neraca['country'] == i) & (neraca['value'] >= 0)]['value'].sum(),

                 neraca[(neraca['country'] == i) & (neraca['value'] <= 0)]['value'].sum()])



sumOfCountry = pd.DataFrame(plot, columns = ['country', 'total','export','import'])

sumOfCountry.head()
sns.barplot(data = sumOfCountry.sort_values(by = ['total'],ascending = True).head(10), y = 'country', x = 'total', orient = 'h')

plt.xticks(rotation = 90)
sns.barplot(data = sumOfCountry.sort_values(by = ['export'],ascending = False).head(10), y = 'country', x = 'export', orient = 'h')

# plt.xticks(rotation = 90)
sns.barplot(data = sumOfCountry.sort_values(by = ['import'],ascending = True).head(10), y = 'country', x = 'import', orient = 'h')

# plt.xticks(rotation = 90)
plot = []

for i in neraca['Commodity'].unique():

    plot.append([i, neraca[neraca['Commodity'] == i]['value'].sum(), 

                 neraca[(neraca['Commodity'] == i) & (neraca['value'] >= 0)]['value'].sum(),

                 neraca[(neraca['Commodity'] == i) & (neraca['value'] <= 0)]['value'].sum()])



sumOfCommodity = pd.DataFrame(plot, columns = ['Commodity', 'total','export','import'])

sumOfCommodity.head()
# plt.figure(figsize = (20,10))

sns.barplot(data = sumOfCommodity.sort_values(by = ['total'],ascending = True).head(10), y = 'Commodity', x = 'total', orient = 'h')

plt.title('Commodity wise total')

sns.barplot(data = sumOfCommodity.sort_values(by = ['export'],ascending = False).head(10), y = 'Commodity', x = 'export', orient = 'h')

plt.title('Commodity wise export')

sns.barplot(data = sumOfCommodity.sort_values(by = ['import'],ascending = True).head(10), y = 'Commodity', x = 'import', orient = 'h')

plt.title('Commodity wise import')
mineral = neraca.groupby(['Commodity']).agg({'value':'sum'})

iterative = mineral.sort_values(by = ['value'], ascending = True).head(10).index

mineralYear = neraca.groupby(['Commodity','year']).agg({'value':'sum'})

mineralYear.sort_values(by = 'value', ascending = True)

plt.figure(figsize = (20,20))

for i in iterative:

    temp = mineralYear.loc[i]

    plt.plot(temp.index, temp['value'], label = i)



plt.legend()

# temp = mineral.loc['MINERAL FUELS, MINERAL OILS AND PRODUCTS OF THEIR DISTILLATION; BITUMINOUS SUBSTANCES; MINERAL WAXES.']

# # temp.columns

# plt.plot(temp.index, temp['value'])
mineral = neraca[neraca['value'] >0].groupby(['Commodity']).agg({'value':'sum'})

iterative = mineral.sort_values(by = ['value'], ascending = False).head(10).index



exportYear = neraca[neraca['value'] >0].groupby(['Commodity','year']).agg({'value':'sum'})

exportYear.sort_values(by = 'value', ascending = True)

plt.figure(figsize = (20,20))

for i in iterative:

    temp = exportYear.loc[i]

    plt.plot(temp.index, temp['value'], label = i)



plt.legend()