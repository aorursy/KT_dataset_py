import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import scipy as sp

import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/avocado-prices/avocado.csv')

original = df.copy()
df.shape
df.info()
df.head()
df = df[['Date', 'AveragePrice', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'type', 'region']]
df.head()
df.describe()
df['Date'] = pd.to_datetime(df['Date'])
df.columns
new_cols = {'4046' : 'Small Haas', '4225' : 'Large Haas', '4770' : 'XLarge Haas'}

df.rename(columns = new_cols, inplace = True)
for i in ['Total Volume', 'Small Haas', 'Large Haas','XLarge Haas', 'Total Bags']:

    df[i] = df[i].astype('int64')
df.info()
df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month
df.head()
summary = pd.pivot_table(index = 'Year', values = ['Total Volume', 'AveragePrice', 'Total Bags'], data = df, 

               aggfunc = {'Total Volume' : sum, 'AveragePrice' : np.mean, 'Total Bags' : sum}).style.background_gradient(cmap = 'Set2')

summary
plt.rcParams['figure.figsize'] = (10, 7)



grouped = df.groupby('Year')['Total Volume'].sum().reset_index()

ax = sns.barplot(x = 'Year', y = 'Total Volume', linewidth = 1, edgecolor = 'k', data = grouped, palette = 'plasma')

for index, row in grouped.iterrows():

    ax.text(row.name, row['Total Volume'], str(round(row['Total Volume'] / 10000000, 2)) + 'Cr.', color = 'k', ha = 'center', va = 'bottom')

plt.title('Total Volume of Avocados Sold by Year', fontsize = 16)

plt.show()
grouped_month = df.groupby('Month')['Total Volume'].sum().reset_index()



ax = sns.barplot(y = 'Month', x = 'Total Volume', data = grouped_month, palette = 'plasma', linewidth = 1, edgecolor = 'k', orient = 'h')

for index, row in grouped_month.iterrows():

    ax.text(row['Total Volume'], row.name, str(round(row['Total Volume'] / 10000000, 2)) + 'Cr.', color = 'k', va = 'bottom')

plt.title('Total Volume of Avocados Sold by Month', fontsize = 16)

plt.show()
grouped_price = df.groupby('Month')['AveragePrice'].mean().reset_index()



ax = sns.lineplot(x = 'Month', y = 'AveragePrice', data = grouped_price, palette = 'plasma', marker = 'v')

plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

for index, row in grouped_price.iterrows():

    ax.text(row.name, row['AveragePrice'], str(round(row['AveragePrice'], 2)) + '$')

plt.axhline((grouped_price['AveragePrice']).min(), color = 'k', label = 'Min. Avg. Price')

plt.axhline((grouped_price['AveragePrice']).max(), color = 'g', label = 'Max. Avg. Price')

plt.title('Avg Price of Avocados by Month', fontsize = 16)

plt.legend(loc = 'best')

plt.show()
grouped_shaas = df.groupby('region')['Small Haas'].sum().sort_values().reset_index()

grouped_lhaas = df.groupby('region')['Large Haas'].sum().sort_values().reset_index()

grouped_xlhaas = df.groupby('region')['XLarge Haas'].sum().sort_values().reset_index()

grouped_reg_total = df.groupby('region')['Total Volume'].sum().sort_values().reset_index()
plt.rcParams['figure.figsize'] = (19, 6)



sns.barplot(x = 'region', y = 'Total Volume', data = grouped_reg_total, palette = 'plasma', linewidth = 1, edgecolor = 'k')

plt.xticks(rotation = 90)

plt.title('Total Avocados Sold by Region', fontsize = 16)

plt.show()
sns.barplot(x = 'region', y = 'Small Haas', data = grouped_shaas, palette = 'plasma', linewidth = 1, edgecolor = 'k')

plt.xticks(rotation = 90)

plt.title('Small Haas Avocados Sold by Region', fontsize = 16)

plt.show()
sns.barplot(x = 'region', y = 'Large Haas', data = grouped_lhaas, palette = 'plasma', linewidth = 1, edgecolor = 'k')

plt.xticks(rotation = 90)

plt.title('Large Haas Avocados Sold by Region', fontsize = 16)

plt.show()
sns.barplot(x = 'region', y = 'XLarge Haas', data = grouped_xlhaas, palette = 'plasma', linewidth = 1, edgecolor = 'k')

plt.xticks(rotation = 90)

plt.title('Extra Large Haas Avocados Sold by Region', fontsize = 16)

plt.show()
plt.rcParams['figure.figsize'] = (10, 7)



grouped_type_total = df.groupby('type')['Total Volume'].sum().reset_index()

ax = sns.barplot(x = 'type', y = 'Total Volume', data = grouped_type_total, palette = 'plasma', linewidth = 1, edgecolor = 'k')

for index, row in grouped_type_total.iterrows():

    ax.text(row.name, row['Total Volume'], str(round(row['Total Volume'] / 10000000, 2)) + 'Cr.', color = 'k', va = 'bottom')

plt.title('Total Volume of Avocados Sold by Type', fontsize = 16)

plt.show()