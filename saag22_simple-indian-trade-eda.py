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
filenames
exports = pd.read_csv(os.path.join(dirname,'2018-2010_export.csv'))

exports.shape
exports.columns
total_value_over_year = exports.groupby(by = ['country','year'])['value'].sum()

total_value_over_year = pd.DataFrame(total_value_over_year)

total_value_over_year.columns
import seaborn as sns

import matplotlib.pyplot as plt
total_value_over_year['value'][5:10].index
df = pd.DataFrame(total_value_over_year['value'])
country = 'U S A'

sns.lineplot(x = df.xs(country).index, y = df.xs(country)['value'], data = df.xs(country))

plt.title('Exports to USA over 2010-18')
countries = exports['country'].unique()
total_value = total_value_over_year.groupby('country')['value'].sum().sort_values()

type(total_value)
total_value = pd.DataFrame(total_value)

print(total_value.index,total_value.columns)
def plot_year_trend(country,df):

    sns.lineplot(x = df.index, y = df['value'], data = df)
countries = total_value.tail(5).index

i = 1

plt.figure(figsize=(12,12))

for country in countries:

    d = df.xs(country)

    plot_year_trend(country,d)

plt.legend(countries)

plt.title('Total value of exports for top 5 countries with maximum value over the period of 2010-18')
total_commodities = exports.groupby('country')['Commodity'].describe().sort_values(by='count',ascending=False)

total_commodities.head(20)
idx_max = exports.groupby(['country'])['value'].transform(max)== exports['value']

most_valuable_export = exports[idx_max]
mvs = most_valuable_export.sort_values(by=['value'],ascending=[False])
mvs.head()
sns.barplot(x = 'country', y = 'value', data = mvs.head(), hue = 'Commodity')

plt.legend()
commodity_export_value = exports.groupby('Commodity')['value'].sum().sort_values(ascending=[False])

commodity_export_value.head()
plt.figure(figsize=(7,7))

color_palette = sns.color_palette(n_colors = 9)

sns.scatterplot(x = 'HSCode', y = 'value', data = mvs, hue = 'year', palette = color_palette)
x = mvs[mvs['value']<1]

x.sample(5)
len(x)
most_valuable_export.groupby('Commodity').describe().sort_values(by=[('HSCode','count')],ascending=[False])
total_value_year = exports.groupby('year')['value'].sum()

total_value_year = pd.DataFrame(total_value_year)
exports.groupby('year').describe()
sns.lineplot(x = total_value_year.index, y = 'value', data = total_value_year)

plt.title('trend of exports for India over the period 2010-18')
total_exports_year = exports.groupby('year')['HSCode'].describe()

sns.lineplot(x = total_exports_year.index, y = 'count', data = total_exports_year)

plt.title('Number of Commodities exported vs year')
cond = (exports['year']==2013) | (exports['year']==2014) | (exports['year']==2015)

exports_2013_15 = exports.loc[cond,:]

exports_2013_15.sample(5)
exports_2013_15.groupby('year').describe()['HSCode']
exports_2010_11 = exports.loc[(exports['year']==2010)|(exports['year']==2011),:]

exports_2010_11.sample(7)
exports_2010_11.groupby('year').describe()