#Import relevant libraries

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns



#Read CSV file

df = pd.read_csv('../input/vgsales.csv')



#Preview the data

df.head(n=5)
#View information on data

df.info()
#Check for null values

df.isnull().any()
#Counting how many NaN values

print ('Year of Release NaN:', df['Year'].isnull().sum())

print ('Publisher NaN:', df['Publisher'].isnull().sum())



#Removing NaN values

df = df.dropna()
#Descriptive statistics

df.loc[:,'NA_Sales':'Global_Sales'].describe()
#Top 25 best-selling video games by global sales copies

df.groupby(['Name']).sum()['Global_Sales'].sort_values(ascending=False)[:25]
#Plotting sales copies by year

global_sales_by_year = df[df['Year'] <= 2016.0].groupby(['Year']).sum()['Global_Sales']

plt.figure(figsize=(10,6))

sns.barplot(y = global_sales_by_year.values, x = global_sales_by_year.index, palette = "PuBu")

plt.xticks(rotation = 70)

plt.title('Copies Sold by Year')

plt.ylabel('Global Copies Sold in Millions')

plt.xlabel('Year')

plt.show()
NA_sales_by_year = df[df['Year'] <= 2016.0].groupby(['Year']).sum()['NA_Sales']

EU_sales_by_year = df[df['Year'] <= 2016.0].groupby(['Year']).sum()['EU_Sales']

JP_sales_by_year = df[df['Year'] <= 2016.0].groupby(['Year']).sum()['JP_Sales']

other_sales_by_year = df[df['Year'] <= 2016.0].groupby(['Year']).sum()['Other_Sales']



the_grid = GridSpec(2, 2)

the_grid.update(left=.01, right=2, wspace=0.1)



plt.subplot(the_grid[0, 0])

sns.barplot(y = NA_sales_by_year.values, x = NA_sales_by_year.index, palette = "PuBu")

plt.title('NA Sales')

plt.axis('off')



plt.subplot(the_grid[0, 1])

sns.barplot(y = EU_sales_by_year.values, x = EU_sales_by_year.index, palette = "PuBu")

plt.title('EU Sales')

plt.axis('off')



plt.subplot(the_grid[1, 0])

sns.barplot(y = JP_sales_by_year.values, x = JP_sales_by_year.index, palette = "PuBu")

plt.title('JP Sales')

plt.axis('off')



plt.subplot(the_grid[1, 1])

sns.barplot(y = other_sales_by_year.values, x = other_sales_by_year.index, palette = "PuBu")

plt.title('Other Sales')

plt.axis('off')



plt.show()
#Plotting global sales copies by genre

sales_by_genre = df.groupby(['Genre']).sum()['Global_Sales'].sort_values(ascending = False)

plt.figure(figsize=(10,6))

sns.barplot(y = sales_by_genre.index, x = sales_by_genre.values, palette = "PuBu_r")

plt.title('Copies Sold by Genre')

plt.xlabel('Global Copies Sold in Millions')

plt.ylabel('Genre')

plt.show()
#Heatmap of sales copies by genre and year

genre_year_table = pd.pivot_table(df[df['Year'] <= 2016.0], values = ['Global_Sales'], index = ['Year'], columns = ['Genre'], aggfunc = 'sum')



plt.figure(figsize=(12,8))

sns.heatmap(genre_year_table['Global_Sales'], annot = True, annot_kws = {"size": 10}, fmt = 'g', cmap = "PuBu")

plt.xticks(rotation = 70)

plt.title('Copies Sold (in Millions) by Year and Genre')

plt.xlabel('Genre')

plt.ylabel('Year')

plt.show()
#Plotting genre sales copies by region

NA_sales_by_genre = df.groupby(['Genre']).sum()['NA_Sales']

EU_sales_by_genre = df.groupby(['Genre']).sum()['EU_Sales']

JP_sales_by_genre = df.groupby(['Genre']).sum()['JP_Sales']

other_sales_by_genre = df.groupby(['Genre']).sum()['Other_Sales']



N = 12

ind = np.arange(N)

width = 0.75



NA_sales_plot = plt.bar(ind,NA_sales_by_genre.values, width, color = '#0570b0', label = "NA Sales")

EU_sales_plot = plt.bar(ind,EU_sales_by_genre.values, width, bottom = NA_sales_by_genre.values, color = '#74a9cf', label = "EU Sales")

JP_sale_plot = plt.bar(ind,JP_sales_by_genre.values, width, bottom = EU_sales_by_genre.values + NA_sales_by_genre.values, color = '#bdc9e1', label = "JP Sales")

other_sales_plot = plt.bar(ind,other_sales_by_genre.values, width, bottom = JP_sales_by_genre.values + EU_sales_by_genre.values + NA_sales_by_genre.values, color = '#f1eef6', label = "Other Sales")



plt.xticks(ind + width/2., NA_sales_by_genre.index, rotation = 70)

plt.title('Copies Sold by Genre and Region')

plt.xlabel('Genre')

plt.ylabel('Copies Sold in Millions')

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)

plt.show()
#Plotting genre sales copies proportions

NA_sales_prop = NA_sales_by_genre / sales_by_genre

EU_sales_prop = EU_sales_by_genre / sales_by_genre

JP_sales_prop = JP_sales_by_genre / sales_by_genre

other_sales_prop = other_sales_by_genre / sales_by_genre



NA_prop = plt.bar(ind,NA_sales_prop.values, width, color = '#0570b0', label = "NA Sales")

EU_prop = plt.bar(ind,EU_sales_prop.values, width, bottom = NA_sales_prop.values, color = '#74a9cf', label = "EU Sales")

JP_prop = plt.bar(ind,JP_sales_prop.values, width, bottom = EU_sales_prop.values + NA_sales_prop.values, color = '#bdc9e1', label = "JP Sales")

other_prop = plt.bar(ind,other_sales_prop.values, width, bottom = JP_sales_prop.values + EU_sales_prop.values + NA_sales_prop.values, color = '#f1eef6', label = "Other Sales")



plt.xticks(ind + width/2., NA_sales_prop.index, rotation = 70)

plt.title('Proportionate Copies Sold by Genre and Region')

plt.xlabel('Genre')

plt.ylabel('Proportion of Copies Sold')

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)

plt.show()
#Heatmap of sales copies by region and genre

genre_region = df.groupby(['Genre']).sum().loc[:,'NA_Sales':'Other_Sales']



plt.figure(figsize=(12,6))

sns.heatmap(genre_region.loc[:,'NA_Sales':'Other_Sales'], annot = True, annot_kws = {"size": 10}, fmt = 'g', cmap = "PuBu")

plt.title('Copies Sold (in Millions) by Genre and Region')

plt.xlabel('Region')

plt.ylabel('Genre')

plt.show()
#Plotting top 10 publishers by number of games released

games_by_publisher = df.groupby(['Publisher']).count()['Name'].sort_values(ascending=False)[:10]

plt.figure(figsize=(10,6))

sns.barplot(y = games_by_publisher.index, x = games_by_publisher.values, palette = "PuBu_r")

plt.title('Number of Games Released by Publisher')

plt.xlabel('Number of Games')

plt.ylabel('Publisher')

plt.show()
#Plotting top 10 publishers by sales copies

sales_by_publisher = df.groupby(['Publisher']).sum()['Global_Sales'].sort_values(ascending = False)[:10]

plt.figure(figsize=(10,6))

sns.barplot(y = sales_by_publisher.index, x = sales_by_publisher.values, palette = "PuBu_r")

plt.title('Copies Sold by Publisher')

plt.xlabel('Global Copies Sold in Millions')

plt.ylabel('Publisher')

plt.show()
#Plotting sales copies per game released by publisher

num_games = df.groupby(['Publisher']).count()['Name']

sales_rev = df.groupby(['Publisher']).sum()['Global_Sales']

revenue_per_game = (sales_rev / num_games).sort_values(ascending = False)[:10]



plt.figure(figsize=(10,6))

sns.barplot(y = revenue_per_game.index, x = revenue_per_game.values, palette = "PuBu_r")

plt.title('Copies Sold per Game Released by Publisher')

plt.xlabel('Copies Sold per Game Released in Millions')

plt.ylabel('Publisher')

plt.show()
#Heatmap of sales copies by publisher and year

top_10_pub = df.groupby(['Publisher']).sum()['Global_Sales'].sort_values(ascending = False)[:10]

publisher_year_table = pd.pivot_table(df[df['Publisher'].isin(top_10_pub.index)][df['Year'] <= 2016.0], values = ['Global_Sales'], index = ['Year'], columns = ['Publisher'], aggfunc = 'sum')



plt.figure(figsize=(12,8))

sns.heatmap(publisher_year_table['Global_Sales'], annot = True, annot_kws = {"size": 10}, fmt = 'g', cmap = "PuBu")

plt.xticks(rotation = 70)

plt.title('Copies Sold (in Millions) by Year and Publisher')

plt.xlabel('Publisher')

plt.ylabel('Year')

plt.show()
df[df['Publisher'] == 'Nintendo'][df['Year'] == 2006][:5]
#Heatmap of sales copies by publisher and genre

publisher_genre_table = pd.pivot_table(df[df['Publisher'].isin(top_10_pub.index)][df['Year'] <= 2016.0], values = ['Global_Sales'], index = ['Genre'], columns = ['Publisher'], aggfunc = 'sum')



plt.figure(figsize=(12,6))

sns.heatmap(publisher_genre_table['Global_Sales'], annot = True, annot_kws = {"size": 10}, fmt = 'g', cmap = "PuBu")

plt.xticks(rotation = 70)

plt.title('Copies Sold (in Millions) by Genre and Publisher (Top 10)')

plt.xlabel('Publisher')

plt.ylabel('Genre')

plt.show()
#Heatmap of sales copies by region and publisher

publisher_region = df[df['Publisher'].isin(top_10_pub.index)].groupby(['Publisher']).sum().loc[:,'NA_Sales':'Other_Sales']



plt.figure(figsize=(12,6))

sns.heatmap(publisher_region.loc[:,'NA_Sales':'Other_Sales'], annot = True, annot_kws = {"size": 10}, fmt = 'g', cmap = "PuBu")

plt.title('Copies Sold (in Millions) by Publisher (Top 10) and Region')

plt.xlabel('Region')

plt.ylabel('Publisher')

plt.show()