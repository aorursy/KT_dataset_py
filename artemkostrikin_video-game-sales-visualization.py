# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the required libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
# Read the csv file

data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
# Let's look at our columns

data.columns
# Check first 20 rows of our data

data.head(20)
#16719 rows and 16 columns

data.shape
# Some statistical analysis of our data

data.describe()

# this gives count, mean, avg etc of all columns containing numerical values
# Forecast for North America

data.NA_Sales.describe()
# Displays the type of values each column in dataset like if the column has float, int, object values,etc

data.info()
# Wii Sports rules

filterr = (data['Global_Sales']==data['Global_Sales'].max())

data['Name'][filterr]  
# The most popular games from the company Electronic Arts'

Publisher_get = data.groupby(['Publisher'])

Publisher_get.get_group('Electronic Arts')
# The most popular games from the company Nintendo

Publisher_get = data.groupby(['Publisher'])

Publisher_get.get_group('Nintendo')
Publisher_get = data.groupby(['Publisher'])

Publisher_get.get_group('Electronic Arts')
sns.pairplot(data)
plt.style.use('fivethirtyeight')

fig = plt.figure(figsize=(25,25))

plt.plot(data['Name'].head(10), data['NA_Sales'].head(10), color='red', label='NA_Sales')

plt.plot(data['Name'].head(10), data['JP_Sales'].head(10), color ='blue', label='JP_Sales')

plt.plot(data['Name'].head(10), data['EU_Sales'].head(10), color='black', label='EU_Sales')

plt.tight_layout()

plt.legend()

plt.xlabel('Famous Games')

plt.ylabel('Sales')

plt.title('Popularity of Famous Games')
# EA vs Nintendo in terms of sales
# filtering out EA sports and Nintendo to compare which company dominates

Nin = (data['Publisher']=='Nintendo')

EA = (data['Publisher']=='Electronic Arts')
# Nintendo sales across various regions

print(data['JP_Sales'][Nin].sum())

print(data['EU_Sales'][Nin].sum())

print(data['NA_Sales'][Nin].sum())

print(data['Global_Sales'][Nin].sum())
# Output for Nintendo

# 458.15             Japan sales of Nintendo

# 419.01             Europe_Sales of Nintendo

# 816.9700000000001  N.America Sales of Nintendo

# 1788.81            Global_Sales of Nintendo
# EA sales across various regions

print(data['JP_Sales'][EA].sum())

print(data['EU_Sales'][EA].sum())

print(data['NA_Sales'][EA].sum())

print(data['Global_Sales'][EA].sum())
# Output for EA

# 14.350000000000001  Japan sales of EA

# 373.90999999999997  Europe_Sales of EA

# 599.5               N.America Sales of EA

# 1116.96             Global_Sales of EA
labels = ['Action', 'Sports', 'Misc', 'Role-Playing', 'Shooter', 'Adventure', 'Racing', 'Platform', 'Simulation', 'Fighting', 'Strategy', 'Puzzle']





plt.style.use('fivethirtyeight')

fig = plt.figure(figsize=(11,11))

plt.pie(list(data['Genre'].value_counts()), autopct='%1.1f%%', labels=labels, wedgeprops={'edgecolor':'black'})

plt.title('Most Popular Genres in Video Games')

plt.tight_layout()

plt.show()
data['Platform'].value_counts().head(10).plot(kind='bar', figsize=(11,5), grid = False, rot=0, color='green')



plt.title('Os 10 Videogames Com Mais titulos lançados')

plt.xlabel('Videogame')

plt.ylabel('Quantidade de jogos lançados')

plt.show()
titulos_lancados = data['Platform'].value_counts()

titulos_lancados.plot()

data['Platform'].value_counts().plot()
y = data.groupby(['Year']).sum()

y = y['Global_Sales']

x = y.index.astype(int)



plt.figure(figsize=(12,8))

ax = sns.barplot(y = y, x = x)

ax.set_xlabel(xlabel='$ Millions', fontsize=16)

ax.set_xticklabels(labels = x, fontsize=12, rotation=50)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)

plt.show();
from wordcloud import WordCloud, STOPWORDS

# Generating the wordcloud with the values under the Platform feature

platform = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1000,

                          height=800

                         ).generate(" ".join(data['Genre'].values))

plt.imshow(platform.recolor( random_state=17), alpha=1)

plt.axis('off')

plt.show()
fig,ax = plt.subplots(figsize=(8,5))

data['Publisher'].value_counts(sort=True).head(10).plot(kind='bar',ax=ax,rot =90)

plt.title('Top 10 Publisher',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('Number of sales',fontsize=15)
x = data.groupby(['Year']).count()

x = x['Global_Sales']

y = x.index.astype(int)



plt.figure(figsize=(12,8))

colors = sns.color_palette("muted")

ax = sns.barplot(y = y, x = x, orient='h', palette=colors)

ax.set_xlabel(xlabel='Number of releases', fontsize=16)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Game Releases Per Year', fontsize=20)

plt.show();
table = data.pivot_table('Global_Sales', index='Genre', columns='Year', aggfunc='sum')

genres = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([genres, sales], axis=1)

data.columns = ['Genre', 'Global Sales']



plt.figure(figsize=(12,8))

ax = sns.pointplot(y = 'Global Sales', x = years, hue='Genre', data=data, size=15, palette='Dark2')

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax.set_title(label='Highest Genre Revenue in $ Millions Per Year', fontsize=20)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

plt.show();
table.columns = table.columns.astype(int)

games = table.idxmax()

sales = table.max()

years = table.columns

data = pd.concat([games, sales], axis=1)

data.columns = ['Game', 'Global Sales']





colors = sns.color_palette("GnBu_d", len(years))

plt.figure(figsize=(12,8))

ax = sns.barplot(y = years , x = 'Global Sales', data=data, orient='h', palette=colors)

ax.set_xlabel(xlabel='Global Sales Per Year', fontsize=16)

ax.set_ylabel(ylabel='Year', fontsize=16)

ax.set_title(label='Highest Revenue Per Game in $ Millions Per Year', fontsize=20)

plt.show();

data