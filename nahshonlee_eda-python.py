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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns
# Load Dataset

df = pd.read_csv('../input/videogamesales/vgsales.csv')

df.head()
df.info()
df.isnull().sum()
# Drop rows with missing Year or missing Publisher

df.dropna(how = 'any', inplace = True)



# Convert Year from float to integer

df['Year'] = df['Year'].astype(int)

df.shape
df.groupby('Year')['Name'].count().sort_values(ascending = False).head(10)
df.groupby('Year')['Name'].count().sort_values().head(10)
# Remove Rows for year 2017 and 2020, as these games are too new to have any useful findings

df = df[~df['Year'].isin([2017, 2020])]
plt.figure(figsize = (8, 6))

ax = sns.countplot(df['Year'], color = '#7FB3D5')

plt.title('No. of Games Released by Year')

plt.xticks(rotation = 90)

plt.ylabel('No. of Games Released')

plt.show()
global_sales_by_year = df.groupby('Year')['Global_Sales'].sum()

plt.figure(figsize = (8, 6))

ax = sns.barplot(x = global_sales_by_year.index, y = global_sales_by_year.values, color = '#EC7063')

plt.xticks(rotation = 90)

plt.ylabel('Global Sales ($m)')

plt.show()
top_100 = df.head(100)

top_100.groupby('Publisher')['Name'].count().sort_values(ascending = False)
top_100.groupby('Publisher')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum().sort_values('Global_Sales', ascending = False)
sns.set_palette('muted')

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (12, 8))



Sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

ax = [ax0, ax1, ax2, ax3]



def plot_sales(Sales_sorted, ax, title):

    sns.barplot(x = Sales_sorted.index, y = Sales_sorted.values, ax = ax, ci = None)

    ax.set(ylabel = 'Total Sales', title = title)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    for i, v in enumerate(Sales_sorted.iteritems()):

        ax.text(i ,v[1], "{:.2f}".format(v[1]), color='m', va ='bottom', rotation=45)



for Sales, ax in zip(Sales, ax):

    Sales_sorted = top_100.groupby('Publisher')[Sales].sum().sort_values(ascending = False)

    plot_sales(Sales_sorted, ax, 'Top 100 Games by Region - '  + Sales)

    

plt.tight_layout()

plt.show()
fig, ax = plt.subplots(1, 1)

Global_Sales = top_100.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending = False)

plot_sales(Global_Sales, ax, 'Top 100 Games Sales - Global')
fig, ax = plt.subplots(1, 1)

Global_Sales = top_100.groupby('Genre')['Global_Sales'].sum().sort_values(ascending = False)

plot_sales(Global_Sales, ax, 'Global Top 100 Games split by Genre')
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (12, 8))



region = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

ax = [ax0, ax1, ax2, ax3]



for region, ax in zip(region, ax):

    Sales_sorted = top_100.groupby('Genre')[region].sum().sort_values(ascending = False)

    plot_sales(Sales_sorted, ax, 'Top 100 Games by Genre - ' + region)

    

plt.tight_layout()

plt.show()
all_sales = df.groupby('Publisher')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum()

all_sales[all_sales['Global_Sales'] > 50].sort_values('Global_Sales', ascending = False)
all_count = df.groupby('Publisher')[['Global_Sales']].count()

all_count[all_count['Global_Sales'] > 100].sort_values('Global_Sales', ascending = False)
sns.set_palette('muted')

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (12, 10))



Sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

ax = [ax0, ax1, ax2, ax3]



for Sales, ax in zip(Sales, ax):

    Sales_sorted = df.groupby('Publisher')[Sales].sum().sort_values(ascending = False)[: 12] # Getting the top 12

    plot_sales(Sales_sorted, ax, 'All Games by Region - ' + Sales)

    

plt.tight_layout()

plt.show()
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (12, 8))



region = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

ax = [ax0, ax1, ax2, ax3]



for region, ax in zip(region, ax):

    Sales_sorted = df.groupby('Genre')[region].sum().sort_values(ascending = False)

    plot_sales(Sales_sorted, ax, 'All Games by Genre - ' + region)

    

plt.tight_layout()

plt.show()
by_platforms = df.groupby(['Year', 'Platform']).sum()

by_platforms = by_platforms.reset_index()

by_platforms.drop(['Rank'], axis = 1, inplace = True)
cmap = sns.cubehelix_palette(start=2.8, rot=.1, as_cmap = True)



fig, ax = plt.subplots(1, 1, figsize = (12, 8))

points = ax.scatter(x = 'Year', y = 'Platform', c = 'Global_Sales', cmap = cmap, data = by_platforms)

fig.colorbar(points)

plt.xlabel('Year')

plt.ylabel('Platform')

plt.title('Global Sales (millions) by Platforms over the Years')

plt.show()
by_platforms.groupby('Platform').sum().drop('Year', axis = 1).sort_values('Global_Sales', ascending = False).head(10)
by_publisher = df.groupby(['Publisher'])['Global_Sales'].sum()

top_5_publisher = by_publisher.sort_values(ascending = False)[:5]

top_5_publisher = top_5_publisher.index.tolist()



by_publisher_genre = df.groupby(['Publisher', 'Genre']).sum()

by_publisher_genre.drop(['Rank', 'Year'], axis = 1, inplace = True)

by_publisher_genre.reset_index(inplace = True)

by_publisher_genre = by_publisher_genre[by_publisher_genre['Publisher'].isin(top_5_publisher)]

by_publisher_genre.head(5)
def sortedgroupedbar(ax, x,y, groupby, data=None, width=0.8, **kwargs):

    sns.set_palette('Set3', n_colors= 12)

    order = np.zeros(len(data))

    df = data.copy()

    for xi in np.unique(df[x].values):

        group = data[df[x] == xi]

        a = group[y].values

        b = sorted(np.arange(len(a)),key=lambda x:a[x],reverse=True)

        c = sorted(np.arange(len(a)),key=lambda x:b[x])

        order[data[x] == xi] = c   

    df["order"] = order

    u, df["ind"] = np.unique(df[x].values, return_inverse=True)

    step = width/len(np.unique(df[groupby].values))

    for xi,grp in df.groupby(groupby):

        ax.bar(grp["ind"]-width/2.+grp["order"]*step+step/2.,

               grp[y],width=step, label=xi, **kwargs)

    ax.legend(title=groupby)

    ax.set_xticks(np.arange(len(u)))

    ax.set_xticklabels(u)

    ax.set_xlabel(x)



fig, ax = plt.subplots(figsize = (12, 8))    

sortedgroupedbar(ax, x="Publisher", y="Global_Sales", groupby="Genre", data=by_publisher_genre)

plt.title('Distribution of Global Sales by Genre for the top 5 Publishers', fontsize = 14)

plt.ylabel('Global Sales (m)')

plt.show()
df.sort_values('Global_Sales', ascending = False).groupby('Genre').head(5).sort_values('Genre')
after2010_df = df[df['Year'] >= 2010]

after2010_df.head()
after2010_top_100 = after2010_df.head(100)

after2010_top_100.groupby('Publisher')['Name'].count().sort_values(ascending = False)
sns.set_palette('muted')

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (12, 10))



Sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

ax = [ax0, ax1, ax2, ax3]



for Sales, ax in zip(Sales, ax):

    Sales_sorted = after2010_top_100.groupby('Publisher')[Sales].sum().sort_values(ascending = False)

    plot_sales(Sales_sorted, ax, 'Games Released After 2010 by Region - ' + Sales)

    

plt.tight_layout()

plt.show()
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (12, 8))



region = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

ax = [ax0, ax1, ax2, ax3]



for region, ax in zip(region, ax):

    Sales_sorted = df.groupby('Genre')[region].sum().sort_values(ascending = False)

    plot_sales(Sales_sorted, ax, 'Games Released after 2010 by Genre - ' + region)

    

plt.tight_layout()

plt.show()
after2010_by_publisher = after2010_df.groupby(['Publisher'])['Global_Sales'].sum()

after2010_top_5_publisher = after2010_by_publisher.sort_values(ascending = False)[:5]

after2010_top_5_publisher = after2010_top_5_publisher.index.tolist()



after2010_by_publisher_genre = after2010_df.groupby(['Publisher', 'Genre']).sum()

after2010_by_publisher_genre.drop(['Rank', 'Year'], axis = 1, inplace = True)

after2010_by_publisher_genre.reset_index(inplace = True)

after2010_by_publisher_genre = after2010_by_publisher_genre[after2010_by_publisher_genre['Publisher'].isin(after2010_top_5_publisher)]

after2010_by_publisher_genre.head(5)
fig, ax = plt.subplots(figsize = (12, 8))    

sortedgroupedbar(ax, x="Publisher", y="Global_Sales", groupby="Genre", data=after2010_by_publisher_genre)

plt.title('Distribution of Global Sales After 2010 by Genre for the top 5 Publishers', fontsize = 14)

plt.ylabel('Global Sales (m)')

plt.show()
after2010_df.sort_values('Global_Sales', ascending = False).groupby('Genre').head(5).sort_values('Genre')