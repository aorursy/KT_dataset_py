import numpy as np

import pandas as pd

import scipy.stats as st

pd.set_option('display.max_columns', None)



import math



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')



import missingno as msno



from sklearn.preprocessing import StandardScaler

from scipy import stats







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

data.head()
drop_row_index = data[data['Year'] > 2015].index

data = data.drop(drop_row_index)
data.shape
data.info()
# data.describe()
# data.describe(include=['object', 'bool'])
data.isnull().sum()
data['Genre'].value_counts()
plt.figure(figsize=(15, 10))

sns.countplot(x="Genre", data=data, order = data['Genre'].value_counts().index)

plt.xticks(rotation=90)
plt.figure(figsize=(15, 10))

sns.countplot(x="Year", data=data, order = data.groupby(by=['Year'])['Name'].count().sort_values(ascending=False).index)

plt.xticks(rotation=90)
plt.figure(figsize=(30, 10))

sns.countplot(x="Year", data=data, hue='Genre', order=data.Year.value_counts().iloc[:5].index)

plt.xticks(size=16, rotation=90)
data_year = data.groupby(by=['Year'])['Global_Sales'].sum()

data_year = data_year.reset_index()

# data_year.sort_values(by=['Global_Sales'], ascending=False)
plt.figure(figsize=(15, 10))

sns.barplot(x="Year", y="Global_Sales", data=data_year)

plt.xticks(rotation=90)
year_max_df = data.groupby(['Year', 'Genre']).size().reset_index(name='count')

year_max_idx = year_max_df.groupby(['Year'])['count'].transform(max) == year_max_df['count']

year_max_genre = year_max_df[year_max_idx].reset_index(drop=True)

year_max_genre = year_max_genre.drop_duplicates(subset=["Year", "count"], keep='last').reset_index(drop=True)

# year_max_genre
genre = year_max_genre['Genre'].values

# genre[0]
plt.figure(figsize=(30, 15))

g = sns.barplot(x='Year', y='count', data=year_max_genre)

index = 0

for value in year_max_genre['count'].values:

#     print(asd)

    g.text(index, value + 5, str(genre[index] + '----' +str(value)), color='#000', size=14, rotation= 90, ha="center")

    index += 1









plt.xticks(rotation=90)

plt.show()
year_sale_dx = data.groupby(by=['Year', 'Genre'])['Global_Sales'].sum().reset_index()

year_sale = year_sale_dx.groupby(by=['Year'])['Global_Sales'].transform(max) == year_sale_dx['Global_Sales']

year_sale_max = year_sale_dx[year_sale].reset_index(drop=True)

# year_sale_max
genre = year_sale_max['Genre']
plt.figure(figsize=(30, 18))

g = sns.barplot(x='Year', y='Global_Sales', data=year_sale_max)

index = 0

for value in year_sale_max['Global_Sales']:

    g.text(index, value + 1, str(genre[index] + '----' +str(round(value, 2))), color='#000', size=14, rotation= 90, ha="center")

    index += 1



plt.xticks(rotation=90)

plt.show()
data_genre = data.groupby(by=['Genre'])['Global_Sales'].sum()

data_genre = data_genre.reset_index()

data_genre = data_genre.sort_values(by=['Global_Sales'], ascending=False)

# data_genre
plt.figure(figsize=(15, 10))

sns.barplot(x="Genre", y="Global_Sales", data=data_genre)

plt.xticks(rotation=90)
data_platform = data.groupby(by=['Platform'])['Global_Sales'].sum()

data_platform = data_platform.reset_index()

data_platform = data_platform.sort_values(by=['Global_Sales'], ascending=False)

# data_platform
plt.figure(figsize=(15, 10))

sns.barplot(x="Platform", y="Global_Sales", data=data_platform)

plt.xticks(rotation=90)
top_game_sale = data.head(20)

top_game_sale = top_game_sale[['Name', 'Year', 'Genre', 'Global_Sales']]

top_game_sale = top_game_sale.sort_values(by=['Global_Sales'], ascending=False)

# top_game_sale
name = top_game_sale['Name']

year = top_game_sale['Year']

y = np.arange(0, 20)
plt.figure(figsize=(30, 18))

g = sns.barplot(x='Name', y='Global_Sales', data=top_game_sale)

index = 0

for value in top_game_sale['Global_Sales']:

    g.text(index, value - 18, name[index], color='#000', size=14, rotation= 90, ha="center")

    index += 1



plt.xticks(y, top_game_sale['Year'], fontsize=14, rotation=90)

plt.xlabel('Release Year')

plt.show()
comp_genre = data[['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]

# comp_genre

comp_map = comp_genre.groupby(by=['Genre']).sum()

# comp_map
plt.figure(figsize=(15, 10))

sns.set(font_scale=1)

sns.heatmap(comp_map, annot=True, fmt = '.1f')



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
comp_table = comp_map.reset_index()

comp_table = pd.melt(comp_table, id_vars=['Genre'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], var_name='Sale_Area', value_name='Sale_Price')

comp_table.head()
plt.figure(figsize=(15, 10))

sns.barplot(x='Genre', y='Sale_Price', hue='Sale_Area', data=comp_table)
comp_platform = data[['Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]

comp_platform.head()
comp_platform = comp_platform.groupby(by=['Platform']).sum().reset_index()
# comp_table = comp_map.reset_index()

comp_table = pd.melt(comp_platform, id_vars=['Platform'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], var_name='Sale_Area', value_name='Sale_Price')

comp_table.head()
plt.figure(figsize=(30, 15))

sns.barplot(x='Platform', y='Sale_Price', hue='Sale_Area', data=comp_table)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
top_publisher = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).head(20)

top_publisher = pd.DataFrame(top_publisher).reset_index()

# top_publisher
plt.figure(figsize=(15, 10))

sns.countplot(x="Publisher", data=data, order = data.groupby(by=['Publisher'])['Year'].count().sort_values(ascending=False).iloc[:20].index)

plt.xticks(rotation=90)
sale_pbl = data[['Publisher', 'Global_Sales']]

sale_pbl = sale_pbl.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(20)

sale_pbl = pd.DataFrame(sale_pbl).reset_index()

# sale_pbl
plt.figure(figsize=(15, 10))

sns.barplot(x='Publisher', y='Global_Sales', data=sale_pbl)

plt.xticks(rotation=90)
comp_publisher = data[['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

comp_publisher.head()
comp_publisher = comp_publisher.groupby(by=['Publisher']).sum().reset_index().sort_values(by=['Global_Sales'], ascending=False)

comp_publisher = comp_publisher.head(20)

# comp_publisher
comp_publisher = pd.melt(comp_publisher, id_vars=['Publisher'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], var_name='Sale_Area', value_name='Sale_Price')

comp_publisher
plt.figure(figsize=(30, 15))

sns.barplot(x='Publisher', y='Sale_Price', hue='Sale_Area', data=comp_publisher)

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.show()
top_publisher =  data[['Year', 'Publisher']]

top_publisher_df = top_publisher.groupby(by=['Year', 'Publisher']).size().reset_index(name='Count')

top_publisher_idx =  top_publisher_df.groupby(by=['Year'])['Count'].transform(max) == top_publisher_df['Count']

top_publisher_count = top_publisher_df[top_publisher_idx].reset_index(drop=True)

top_publisher_count  = top_publisher_count.drop_duplicates(subset=["Year", "Count"], keep='last').reset_index(drop=True)

# top_publisher_count
publisher= top_publisher_count['Publisher']
plt.figure(figsize=(30, 15))

g = sns.barplot(x='Year', y='Count', data=top_publisher_count)

index = 0

for value in top_publisher_count['Count'].values:

#     print(asd)

    g.text(index, value + 5, str(publisher[index] + '----' +str(value)), color='#000', size=14, rotation= 90, ha="center")

    index += 1









plt.xticks(rotation=90)

plt.show()
# data.head()
top_sale_reg = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]

# pd.DataFrame(top_sale_reg.sum(), columns=['a', 'b'])

top_sale_reg = top_sale_reg.sum().reset_index()

top_sale_reg = top_sale_reg.rename(columns={"index": "region", 0: "sale"})

top_sale_reg
plt.figure(figsize=(12, 8))

sns.barplot(x='region', y='sale', data = top_sale_reg)
labels = top_sale_reg['region']

sizes = top_sale_reg['sale']
plt.figure(figsize=(10, 8))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
# sns.distplot(data['NA_Sales'],  kde=False, fit=stats.gamma);

# sns.distplot(data['EU_Sales'],  kde=False, fit=stats.gamma);

plt.figure(figsize=(25,30))

sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

for i, column in enumerate(sales_columns):

    plt.subplot(3,2,i+1)

    sns.distplot(data[column], bins=20, kde=False, fit=stats.gamma)
data_hist_log = data.copy()
data_hist_log = data_hist_log[data_hist_log.NA_Sales != 0]

data_hist_log = data_hist_log[data_hist_log.EU_Sales != 0]

data_hist_log = data_hist_log[data_hist_log.Other_Sales != 0]

data_hist_log = data_hist_log[data_hist_log.JP_Sales != 0]

data_hist_log = data_hist_log[data_hist_log.Global_Sales != 0]
plt.figure(figsize=(25,30))

sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

for i, column in enumerate(sales_columns):

    plt.subplot(3,2,i+1)

    sns.distplot(np.log(data_hist_log[column]), bins=20, kde=False, fit=stats.gamma)
plt.figure(figsize=(13,10))

sns.heatmap(data.corr(), cmap = "Blues", annot=True, linewidth=3)
data_pair = data.loc[:,["Year","Platform", "Genre", "NA_Sales","EU_Sales", "Other_Sales"]]

data_pair
sns.pairplot(data_pair, hue='Genre')
data_pair_log = data_pair.copy()
sale_columns = ['NA_Sales', 'EU_Sales', 'Other_Sales']
# for column in sale_columns:

#     if 0 in data[column].unique():

#         pass

#     else:

#         data_pair_log[column] = np.log(data_pair_log[column])

# #         data_pair_log.head()
data_pair_log = data_pair_log[data_pair_log.NA_Sales != 0]

data_pair_log = data_pair_log[data_pair_log.EU_Sales != 0]

data_pair_log = data_pair_log[data_pair_log.Other_Sales != 0]
data_pair_log
data_pair_log['NA_Sales'] = np.log(data_pair_log['NA_Sales']);

data_pair_log['EU_Sales'] = np.log(data_pair_log['EU_Sales']);

data_pair_log['Other_Sales'] = np.log(data_pair_log['Other_Sales']);
# sns.pairplot(data_pair_log, hue='Genre',  diag_kind = 'kde',

#              plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

#              size = 4)



sns.pairplot(data_pair_log, hue='Genre',  palette="husl")