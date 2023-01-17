import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set()

pd.set_option('display.max_rows',16)

pd.set_option('display.max_columns',100)

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# Any results you write to the current directory are saved as output.
games = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

games.head()
games.shape
len(games.Name.unique())
games.isnull().sum()
games.dropna(axis='rows', subset=['Year_of_Release'], inplace=True)

games.Year_of_Release = games.Year_of_Release.astype(np.int64)
multiple_platform_title = games.groupby('Name').agg({'Name':lambda x : len(x) if (len(x) > 1) else None}).dropna()

multiple_platform_title.Name = multiple_platform_title.Name.astype(np.int64)

multiple_platform_title = multiple_platform_title.sort_values(['Name'], ascending=False)

multiple_platform_title.columns = ['Platform_count']

multiple_platform_title
cols = ['Name','Platform','Year_of_Release','Publisher']

games.loc[games.Name=='Need for Speed: Most Wanted', cols].sort_values(['Year_of_Release'])
platform_title = games.Platform.value_counts()



plt.subplots(figsize=(8,7))

ax = sns.barplot(x=platform_title , y=platform_title .index, palette='cubehelix')

ax.set_title('Most Number of Titles per Platform', color='red', alpha=0.5, size=25)

ax.set_xlabel('Total Titles', color='green', alpha=0.5, size=30)

ax.set_ylabel('Platform', color='green', alpha=0.5, size=30)
# games based on genre

games_by_genre = games.groupby('Genre').agg({'Genre':len}).sort_values('Genre')

plt.subplots(figsize=(10,7))

ax = sns.pointplot(x=games_by_genre.index, y=games_by_genre.Genre)

ax.set_title('Total Number of Games by Genre', color='blue', size=25, alpha=0.5)

ax.set_xlabel('Genre', color='green', size=25, alpha=0.5)

ax.set_ylabel('Total Number of Games', color='green', size=25, alpha=0.5)
global_sales_by_genre = games.groupby('Genre').agg({'Global_Sales':np.sum}).sort_values('Global_Sales')

plt.subplots(figsize=(11,7))

ax = sns.barplot(x=global_sales_by_genre.index, y=global_sales_by_genre.Global_Sales)

ax.set_title('Total Global Sales of Games by Genre (1980-2016)', color='blue', size=25, alpha=0.5)

ax.set_xlabel('Genre', color='green', size=25, alpha=0.5)

ax.set_ylabel('Total Global Sales', color='green', size=25, alpha=0.5)
global_sales_publisher = games.pivot_table(index=['Publisher'], values=['NA_Sales','EU_Sales','JP_Sales'], 

                                           aggfunc=np.sum).sort_values(['NA_Sales'], ascending=False)

global_sales_publisher = global_sales_publisher[['NA_Sales','EU_Sales','JP_Sales']]

ax = global_sales_publisher.iloc[0:10,:].plot(kind='bar', stacked=True, grid=False)

ax.set_title('Top 10 Publishers by Sales', size=25, color='blue', alpha=0.5)

ax.set_xlabel('Publisher', size=25, color='green', alpha=0.5)

ax.set_ylabel('Sales', size=25, color='green', alpha=0.5)
titles_by_year = games.groupby(['Year_of_Release']).agg({'Name':lambda x : len(x) if (len(x) > 1) else None}).dropna()

plt.subplots(figsize=(11,7))

ax = sns.pointplot(x=titles_by_year.index, y=titles_by_year.Name)

g = ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.set_title('Number of Titles by Year of Release', size=25, color='blue', alpha=0.5)

ax.set_xlabel('Year of Release', size=25, color='red', alpha=0.5)

ax.set_ylabel('Number of Titles', size=25, color='red', alpha=0.5)
platform_after_2000 = games.loc[(games.Platform == 'PS2') | (games.Platform == 'XB') | (games.Platform == 'GC'),:]

platform_after_2000 = platform_after_2000.groupby(['Platform']).agg(np.sum)



ax = sns.heatmap(platform_after_2000.iloc[:, 1:5])

ax.set_yticklabels(['XBOX', 'PS2', 'Game Cube'])

ax.set_xticklabels(['NA Sales', 'EU Sales', 'JP Sales', 'Other Sales'])

ax.set_xlabel('Region Sales', size=25, color='blue', alpha=0.5)

ax.set_ylabel('Platform', size=25, color='blue', alpha=0.5)
platform_after_2000_2009 = games.loc[((games.Platform == 'PS2') | (games.Platform == 'XB') | (games.Platform == 'GC')) & 

                                     ((games.Year_of_Release >= 2001) & (games.Year_of_Release <= 2009)),:]

platform_after_2000_2009 = platform_after_2000_2009.groupby(['Platform']).agg(np.sum)



ax = sns.heatmap(platform_after_2000_2009.iloc[:, 1:5])

ax.set_yticklabels(['XBOX', 'PS2', 'Game Cube'])

ax.set_xticklabels(['NA Sales', 'EU Sales', 'JP Sales', 'Other Sales'])

ax.set_xlabel('Region Sales', size=25, color='blue', alpha=0.5)

ax.set_ylabel('Platform', size=25, color='blue', alpha=0.5)