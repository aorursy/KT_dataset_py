# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import squarify

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

df.head()
# converting proper data type

df['Date'] = pd.to_datetime(df.Date, format = "%Y/%m/%d")

df['month'] = df.Date.dt.month

df['day'] = df.Date.dt.day

df.info()
# inspecting categorical variables

df = df.drop('Unnamed: 0', axis = 1)

print('Avocado Region List\n', df.region.unique())

print('-' * 40)

print('Avocado Type List\n', df.type.unique())
# manipulating data (seperate into three datasets according to region)

df_US = df[df.region == 'TotalUS']

df = df.drop(df_US.index, axis = 0)



region = ['West', 'Midsouth', 'Northeast', 'Southcentral', 'Southeast']

df_region = df[(df.region == 'Midsouth') | (df.region == 'Northeast') |

               (df.region == 'SouthCentral') | (df.region == 'Southeast') |

               (df.region == 'West')]



df = df.drop(df_region.index, axis = 0)



# checking regions in datasets 

print('US Dataset Region:\n', df_US.region.unique())

print('-'*40, '\nRegion Dataset Region:\n', df_region.region.unique())

print('-'*40, '\nOriginal Dataset Region:\n', df.region.unique())
print('---- US Avocado Statistics ----\n'), display(df_US.describe().T)

print('---- Regional Avocado Statistics ----\n'), display(df_US.describe().T)

print('---- Avocado Statistics ----\n'), display(df_US.describe().T)
# Price Overview by Categorical Variables

region5 = df_region[['region', 'AveragePrice']].groupby('region').agg('mean').sort_values(

                        by = 'AveragePrice', ascending = False).reset_index()



print('Top 5 Avg.Price Overview by Big Region')

display(region5.head(5))

print('-' * 35)



check_list = ['type', 'year', 'month', 'day', 'region']

for cat in check_list:

    top5 = df[[cat, 'AveragePrice']].groupby(cat).agg('mean').sort_values(

                        by = 'AveragePrice', ascending = False).reset_index().head(5)

    print('Top {} Avg.Price Overview by {}'.format(top5.shape[0], str.capitalize(cat)))

    display(top5)

    print('-' * 35)
# overview on the dataset

sns.pairplot(df.drop(['region', 'type'], axis = 1))

plt.show()
# correlation overview on the dataset

plt.figure(figsize = (10, 8))

sns.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = 'Blues')

plt.title('Correlation among All Variables\n', fontsize = 14)

plt.show()
sns.set_style('darkgrid')

plt.figure(figsize = (15, 8))

sns.lineplot('Date', 'AveragePrice', data = df_US, hue = 'type', alpha = .8)

plt.title('US Price Overview', fontsize = 14)

plt.show()
g = sns.FacetGrid(df_region, col = 'type', row = 'year', hue = 'region', 

                  height = 3, aspect = 2, palette = 'RdBu_r')

g.map(sns.lineplot, 'month', 'AveragePrice')

g.add_legend()

plt.show()
sns.factorplot(x = 'AveragePrice', y = 'region', data = df,

              hue = 'year', size = 15, aspect = 0.7, palette = 'RdBu_r', join = False)

plt.title('Yearly Average Price Overview on City', fontsize = 12)

plt.ylabel('')

plt.show()
plt.figure(figsize = (18, 8))

ax = sns.lineplot('Date', 'Total Volume', data = df, label = 'Volume', legend = False)

ax2 = plt.twinx()

sns.lineplot('Date', 'Total Bags', data = df, 

             color = 'orange', label = 'Bags', legend = False, ax = ax2)

ax.figure.legend()

plt.show()
fig, axes = plt.subplots(1, 2, figsize = (15, 5))

g = sns.boxplot(x = 'Total Volume', y = 'region', data = df_region, 

                hue = 'type', ax = axes[0])

g.set(title = 'Total Volume by Region', ylabel = ' ')



g = sns.boxplot(x = 'Total Bags', y = 'region', data = df_region, 

                hue = 'type', ax = axes[1])

g.set(title = 'Total Bags by Region', ylabel = '')

plt.show()
# grouping the data by region with desc order by total volume

volume_order = df.groupby('region')['Total Volume'].sum(

                        ).sort_values(ascending = False).reset_index()



# setting the tree map

cmap = matplotlib.cm.Blues

volume_values = [i for i in range(volume_order.shape[0])]

norm = matplotlib.colors.Normalize(vmin = min(volume_values), vmax = max(volume_values))

colors = [cmap(norm(value)) for value in volume_values][::-1]



# plotting the map

plt.figure(figsize = (20, 12))

squarify.plot(sizes = volume_order['Total Volume'], alpha = 0.8,

              label = volume_order.region, color = colors)

plt.title('Region Total Volume Tree Map', fontsize = 20)

plt.axis('off')

plt.show()
# grouping the data by region with desc order by total bags

bag_order = df.groupby('region')['Total Bags'].sum(

                        ).sort_values(ascending = False).reset_index()



cmap = matplotlib.cm.Reds

bag_values = [i for i in range(bag_order.shape[0])]

norm = matplotlib.colors.Normalize(vmin = min(bag_values), vmax = max(bag_values))

colors = [cmap(norm(value)) for value in bag_values][::-1]



plt.figure(figsize = (20, 12))

squarify.plot(sizes = bag_order['Total Bags'], alpha = 0.8,

              label = bag_order.region, color = colors)

plt.title('Region Total Bag Tree Map', fontsize = 20)

plt.axis('off')

plt.show()
# yearly total bags VS total volume overview

g = sns.FacetGrid(df_region, col = 'type', row = 'year', palette = 'RdBu_r',

                      hue = 'region',  height = 3.5, aspect = 2)

g.map(sns.scatterplot, 'Total Volume', 'Total Bags')

g.add_legend()

plt.show()
plt.style.use('seaborn')



# grouping & summing the data by date / plotting the area based on types

fig, axes = plt.subplots(2, 1, figsize = (15, 10))

df[['4046', '4225', '4770', 'Date']].groupby('Date').sum().plot(kind = 'area', 

                        title = 'Volume Type Overview', ax = axes[0])

df[['Small Bags', 'Large Bags', 'XLarge Bags', 'Date']].groupby('Date').sum().plot(

                        kind = 'area',title = 'Bag Type Overview', ax = axes[1])

plt.tight_layout()

plt.show()
# grouping volume types by big region sorted with desc order by total volume 

vol_region = df_region[['4046', '4225', '4770', 'Total Volume', 'year', 'region']].groupby(

                ['year','region']).sum().sort_values(

                ['year', 'Total Volume'], ascending = False).reset_index().drop(

                'Total Volume', axis = 1)



fig, axes = plt.subplots(2, 2, figsize = (12, 8))



for ax, year in zip(axes.flatten(), vol_region.year.unique()[::-1]):

    vol_region.loc[vol_region.year == year][['region', '4046', '4225', '4770']].plot(

                                        x = 'region', kind = 'bar', stacked = True, ax = ax,

                                        title = 'Regional Volume Details in {}'.format(year))

    ax.set(xlabel = '')

    ax.tick_params(axis = 'x', labelrotation = 45)



plt.tight_layout()

plt.show()
# grouping bag types by big region sorted with desc order by total bags

bag_region = df_region[['Small Bags', 'Large Bags', 'XLarge Bags', \

                        'Total Bags', 'year', 'region']].groupby(

                        ['year','region']).sum().sort_values(

                        ['year', 'Total Bags'], ascending = False).reset_index().drop(

                        'Total Bags', axis = 1)



fig, axes = plt.subplots(2, 2, figsize = (12, 8))



for ax, year in zip(axes.flatten(), bag_region.year.unique()[::-1]):

    bag_region.loc[bag_region.year == year][['region', \

                        'Small Bags', 'Large Bags', 'XLarge Bags']].plot(

                        x = 'region', kind = 'bar', stacked = True, ax = ax,

                        title = 'Regional Bags Details in {}'.format(year))

    ax.set(xlabel = '')

    ax.tick_params(axis = 'x', labelrotation = 45)



plt.tight_layout()

plt.show()
# grouping volume types by region sorted with desc order by total volume 

vol_all = df[['4046', '4225', '4770', 'Total Volume', \

            'year', 'region']].groupby('region').sum().sort_values(

            'Total Volume').reset_index().drop('Total Volume', axis = 1)



vol_all.plot(x = 'region', kind = 'barh', stacked = True,

                 title = 'City Volume Details', figsize = (12, 15))

plt.ylabel('')

plt.show()
# grouping bag types by region sorted with desc order by total bags

bag_all = df[['Small Bags', 'Large Bags', 'XLarge Bags','Total Bags', \

            'region']].groupby('region').sum().sort_values(

            'Total Bags').reset_index().drop('Total Bags', axis = 1)



bag_all.plot(x = 'region', kind = 'barh', stacked = True,

                 title = 'City Bags Details', figsize = (12, 15))

plt.ylabel('')

plt.show()