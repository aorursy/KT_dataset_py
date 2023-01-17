# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Plotting - matplotlib, seaborn



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/vgsales.csv")

df.head()
df.info()
print('--- BASIC STATS ---')



# Years covered?

print('Dataset has games from %d' %df['Year'].min() + ' - %d' %df['Year'].max())



# How many unique games?

print('Number of Unique Games listed: ' + str(len(df['Name'].unique())))



# How many game publishers?

print('Number of Publishers listed: ' + str(len(df['Publisher'].unique())))



# How many game platforms?

print ('Number of Platforms listed: ' + str(len(df['Platform'].unique())))



# How many game genres?

print('Number of Genres listed: ' + str(len(df['Genre'].unique())))

print(df['Genre'].unique())
# Exactly how many NaN (missing) values?



print('Amount of NaN values for each column:')

for column in df.columns:

    print(column + ':' + str(len(df[df[column].isnull()])))
correlation = df.corr()

correlation
# Let's visualize the correlations with seaborn

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

plt.title('Correlation between different features')
# Top 25 Publishers by Global Sales



publishers = df.groupby(['Publisher']).sum()

top25_publishers = publishers.sort_values(by='Global_Sales', ascending=False)[:25]

top25_publishers



plt.figure(figsize=(8,6))

sns.barplot(y=top25_publishers.index, x=top25_publishers.Global_Sales)

plt.ylabel("Publisher")

plt.xlabel("Global Sales")

plt.title('Top 25 Publishers by Global Sales')

plt.show()
# Top 25 Publishers with most releases



mostgames_publisher = pd.crosstab(df.Publisher, df.Name)

mostgames_sum = mostgames_publisher.sum(axis=1)

top25_games = mostgames_sum.sort_values(ascending=False)[:25]

plt.figure(figsize=(8,6))

sns.barplot(y=top25_games.index, x=top25_games.values, orient="h")

plt.ylabel("Year")

plt.xlabel("Number of Games")

plt.title('Top 25 Publishers by # of Games Released')

plt.show()
# Top 25 Publishers by Avg. Sales per Game Release

# Take the Top 25 Publishers by Global Sales and order them based on the average sales per game release



sales_per_game = (top25_publishers['Global_Sales']/mostgames_sum[top25_publishers.index]).sort_values(ascending=False)[:25]

sales_per_game

plt.figure(figsize=(8,6))

sns.barplot(y=sales_per_game.index, x=sales_per_game.values, orient="h")

plt.ylabel("Publishers")

plt.xlabel("Avg. Sales per Game Release")

plt.title('Top 25 Publishers Globally by Avg. Sales per Game Release')

plt.show()
def sum_globalsales(keyword):

    '''

    Finds the total amount of 

    Global Sales for a series of

    games with the same starting word/phrase

    '''

    total_sales = 0

    print("'" + keyword + "'" + ' Series')

    print('---TITLES---')

    for title in df['Name'].unique():   # list of unique game titles to avoid duplication below

        if title.startswith(keyword):

            group = df[df.Name == title]   # accounts for games released on multiple platforms

            for key in group.index:

                sales = df.iloc[key]['Global_Sales']

                print(title + ': ' + str(sales) + ' [' + df.iloc[key]['Platform'] + ']')

                total_sales += sales

    print('-'*len('---TITLES---'))

    print(total_sales)
# The Legend of Zelda



sum_globalsales('The Legend of Zelda')
# Gears of War



sum_globalsales('Gears of War')
# Halo



sum_globalsales('Halo')
data = (df.groupby('Genre').sum())



# Top Genres Globally

plt.figure(figsize=(12,6))

sns.barplot(y=data.index, x=data.Global_Sales, orient="h")

plt.ylabel("Genre")

plt.xlabel("Global Sales")

plt.title('Top Genres Globally')



# Top Genres for NA, EU, JP, Other

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(16,5))

sns.barplot(y=data.index, x=data.NA_Sales, orient="h", ax=axis1)

sns.barplot(y=data.index, x=data.EU_Sales, orient="h", ax=axis2)



fig, (axis1, axis2) = plt.subplots(1,2,figsize=(16,5))

sns.barplot(y=data.index, x=data.JP_Sales, orient="h", ax=axis1)

sns.barplot(y=data.index, x=data.Other_Sales, orient="h", ax=axis2)
df[df.Genre == 'Adventure']['Name'].head(10)
genreGame = pd.crosstab(df.Genre, df.Name)

genreGameSum = genreGame.sum(axis=1).sort_values(ascending=False)

plt.figure(figsize=(12,6))

sns.barplot(y=genreGameSum.index, x=genreGameSum.values, orient="h")

plt.ylabel("Genre")

plt.xlabel("Number of Games")

plt.title("Genres with the Most Releases")

plt.show()