import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings; warnings.filterwarnings('ignore')

%matplotlib inline



pd.options.display.max_rows = 100

plt.style.use('seaborn-white')
# Load the dataset

data = pd.read_csv('../input/videogamesales/vgsales.csv')

data.head()
data.info()
data.describe()
data.describe(include='O')
quantitative = [f for f in data.columns if data.dtypes[f] != 'object']

qualitative = [f for f in data.columns if data.dtypes[f] == 'object']



print ('Quantitiative features:', quantitative)

print ('\nQualitative features:', qualitative)
# Find columns with missing data

missing = data.isnull().sum()

missing = missing[missing > 0]

print (missing)
# Isolate missing data of 'Year' to look for any possible pattern

missing_year = data[data.Year.isna()]

missing_year.describe(include='all')
missing_year['Publisher'].value_counts(dropna=False).head()
# Isolate missing data of 'Publisher' (NaN + 'Unknown')

missing_publisher = data.loc[(data.Publisher.isna()) | (data.Publisher.str.contains('Unknown'))]

missing_publisher['Year'].value_counts(dropna=False).head()
data['Publisher'].fillna('Unknown', inplace=True)

data['Publisher'].isna().sum()
# Examine consistency between 'Global_Sales' and the summation of all regional sales

data['Sum_Sales'] = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales' ]].sum(axis=1)

data['Sum_Diff_abs'] = data['Sum_Sales'] - data['Global_Sales']

data['Sum_Diff_Perc'] = data['Sum_Diff_abs'] / data['Global_Sales']

print('Sorting by absolute different values:')

print(data[['Global_Sales', 'Sum_Sales', 'Sum_Diff_abs', 'Sum_Diff_Perc']].sort_values(by='Sum_Diff_abs', ascending=False))

print('\nSorting by percentage:')

print(data[['Global_Sales', 'Sum_Sales', 'Sum_Diff_abs', 'Sum_Diff_Perc']].sort_values(by='Sum_Diff_Perc', ascending=False))
data[quantitative].hist(bins=20, figsize=(18,12))

plt.show()
# Use z-score to check up the low end and the high end of the distribution of 'Global_Sales'

from sklearn.preprocessing import StandardScaler



global_scaled = StandardScaler().fit_transform(data['Global_Sales'][:, np.newaxis]) # z-score

low_range = global_scaled[global_scaled[:, 0].argsort()][:10]

high_range = global_scaled[global_scaled[:, 0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
# The number of games of high outer range whose z-scores are greater than 3

high_sales_games = global_scaled[global_scaled[:,0] > 3].shape[0]

# The revenue of sales of high outer range whose z-scores are greater than 3

high_sales = data['Global_Sales'].sort_values(ascending=False)[:high_sales_games]



ratio_number = high_sales_games / data.shape[0]

ratio_sales = np.sum(high_sales) / np.sum(data['Global_Sales'])

print('ratio of numbers of high sales games vs total number of games: %.1f%%' %(100 * ratio_number))

print('ratio of revenue of high sales games vs total sales revenue: %.1f%%' %(100 * ratio_sales))
# Plot correlation heatmap

corr = data[quantitative].corr()

plt.figure(figsize=(8,6))

sns.heatmap(corr, cmap='RdYlGn', center=0, annot=True)

plt.show()
# Plot sales trends on following categorical variables - 'Platform', 'Genre', 'Year'

cat_columns = ['Year', 'Platform', 'Genre']  

sales_columns = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']



for cat in cat_columns:

    sales_groups = data.groupby(cat)[sales_columns].sum()

    if cat != 'Year':

        sales_groups = sales_groups.sort_values(by='Global_Sales', ascending=False)

    x = sales_groups.index.values



    plt.figure(figsize=(8,6))

    

    for sales in sales_columns:

        plt.plot(x, sales_groups[sales], marker='.', linewidth=2.5, alpha=.6, label=sales)

    

    plt.title('Regional Sales On %s' % cat, fontsize=16)

    plt.xticks(rotation=60, fontsize=12)

    plt.yticks(fontsize=12)

    plt.ylabel('Sales Revenue(US$ in millions)', fontsize=14)

    plt.grid(linestyle='--', alpha=0.5)

    plt.legend(fontsize=12)



plt.show()
## Create a new dataframe for explanatory analysis

data_analy = data.copy()



# Create a new categorical column of every 5 years

ranges = [1979, 1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019] # for range: (1979, 1984], etc.

group_names = ['80-84', '85-89', '90-94', '95-99', '00-04', '05-09', '10-14', '15-19']

data_analy['Period'] = pd.cut(data_analy['Year'], bins=ranges, labels=group_names)



data_analy.head()
# Prepare data for plot

regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

regional_sales = data_analy[regions].sum(axis=0)



# Plot

plt.figure(figsize=(10,8))

plt.pie(regional_sales, labels=regional_sales.index.values, autopct='%.1f%%', pctdistance=.6,

        startangle=-90, textprops=dict(size=16))



plt.title('The Ratio of Sales Revenue for Each Region', fontsize=20)

plt.show()
# Prepare data by grouping genres, then sum up sales of each region

genre_sales = data_analy[regions + ['Genre']].groupby('Genre').sum()

genre_sales = genre_sales.sort_values(by='NA_Sales', ascending=False)



# Plot

fig, ax = plt.subplots(2, 2, figsize=(18, 16))

ax = ax.ravel() # flatten to 1D array

labels = genre_sales.index.values

explode = (0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1)



for idx, ax in enumerate(ax):

    region = regions[idx]

    x = genre_sales.loc[:, region]

    ax.pie(x, labels=labels, autopct='%.1f%%', pctdistance=.8, startangle=20, explode=explode, 

           textprops=dict(size=13, style='italic', weight='bold'))

    

    ax.set_title('%s By Each Genre' % region, fontsize=18)

    

plt.show()
# Grouping by each genre and each period, then count numbers

genre_games = data_analy[['Genre', 'Period']].groupby(['Genre', 'Period']).size()

genre_games = genre_games.reset_index() # Convert a multiindex Series to a dataframe

genre_games = genre_games.rename(columns={0:'Games'})



# Grid plot for each genre

def lineplot(x, y, **kwargs):

    plt.plot(x, y, 'ko-', alpha=0.7)

    plt.grid(linestyle='--', alpha=0.5)  

    plt.fill_between(x, y, 0, facecolor='orange', alpha=0.2)

    for tx, ty in zip(x, y):

        plt.text(tx, ty+25, ty, horizontalalignment='right', 

                 fontdict={'style':'italic'})

        

g = sns.FacetGrid(genre_games, col='Genre', sharex=False, sharey=False, col_wrap=3, size=4,

                 col_order=genre_sales.index.values)

g = (g.map(lineplot, 'Period', 'Games').set(ylim=(0, 1200), xlabel='Years', ylabel='Number of Games')

    .set_xticklabels(genre_games.Period, fontsize=10, rotation=45).fig.subplots_adjust(hspace=0.3))

    

plt.show()
# Prepare data by grouping platforms, then sum up sales of each region

platform_sales = data_analy[regions + ['Platform']].groupby('Platform').sum()

platform_sales = platform_sales.sort_values(by='NA_Sales')



# Plot

fig, ax = plt.subplots(figsize=(10,16))

ax.hlines(y=platform_sales.index, xmin=0, xmax=650, color='gray', alpha=0.8, linewidth=1,

          linestyle='dashdot')

for region in regions:

    ax.scatter(x=platform_sales[region], y=platform_sales.index, s=75, alpha=0.7, label=region)



# Decorate

ax.set_title('Comparison of Regional Sales on Each Platform (US$ in Milli)', fontdict={'size': 20})

ax.xaxis.set_ticks_position('top')

ax.set_xlim(-20, 670)

plt.xticks(np.arange(0, 660, 50), fontsize=12)

plt.yticks(fontsize=12)



plt.legend(loc=(0.98, 0.9), fontsize=14)

plt.grid(axis='x', linestyle='--', alpha=0.4)

plt.show()
# Prepare data for the proportion of each platform in each region

for region in regions:

    platform_sales[region + '_Ratio'] = platform_sales[region]/np.sum(platform_sales[region])

platform_sales = platform_sales.sort_values(by='NA_Sales', ascending=False)



# Plot stacked bar chart for the proportions

xlabels = platform_sales.index.values

x = np.arange(len(xlabels))

width = 0.6 # width of bars

ratios = ['NA_Sales_Ratio', 'EU_Sales_Ratio', 'JP_Sales_Ratio', 'Other_Sales_Ratio']

bottom = 0



fig, ax = plt.subplots(figsize=(16,8))

for ratio in ratios:

    y = platform_sales[ratio]

    ax.bar(x, y, width, bottom=bottom, edgecolor='white', alpha=.7, label=ratio)

    bottom += y



# Decoration

plt.title('The Proportion of Sales of Each Platform in Each Regional Market', fontsize=20)

plt.yticks(np.arange(0, 0.66, 0.05), fontsize=14)

plt.xticks(x, xlabels, rotation=90, fontsize=14)

plt.xlabel('Platforms', fontsize=16)

ax.set_facecolor('#E8E8E8')

plt.grid(color='white', axis='y', linestyle='--')



plt.legend(loc='upper center', bbox_to_anchor=(0.6, 0.95), ncol=4,

           fancybox=True, shadow=True, fontsize=14)

plt.show()
# Grouping by each platform and each period, then counting numbers

platform_games = data_analy[['Platform', 'Period']].groupby(['Platform', 'Period']).size()

platform_games = platform_games.reset_index() # Convert a multiindex Series to a dataframe

platform_games = platform_games.rename(columns={0:'Games'})



# Grid plot for each platform

def lineplot(x, y, **kwargs):

    plt.plot(x, y, 'ko-', alpha=0.7)

    plt.grid(linestyle='--', alpha=0.5)  

    plt.fill_between(x, y, 0, facecolor='orange', alpha=0.2)

    for tx, ty in zip(x, y):

        plt.text(tx, ty+25, ty, horizontalalignment='right', 

                 fontdict={'style':'italic'})

        

g = sns.FacetGrid(platform_games, col='Platform', sharex=False, sharey=False, col_wrap=3, size=4,

                 col_order=platform_sales.index.values)

g = (g.map(lineplot, 'Period', 'Games').set(ylim=(0, 1600), xlabel='Years', ylabel='Number of Games')

    .set_xticklabels(genre_games.Period, fontsize=10, rotation=45).fig.subplots_adjust(hspace=0.3))

    

plt.show()
# Sorting out the top 100 games based on sales for each region

data_100 = []

s = [] # sequence of the global 'Rank' for each region's top 100 games

for i, region in enumerate(regions):

    data_100.append(data_analy.sort_values(by=region, ascending=False)[:100])

    s.append(data_100[i].Rank)

    

# Plot the difference of top 100 games between each pair of regions using heatmap

fig, ax = plt.subplots(figsize=(8, 6))

corr = pd.DataFrame(np.zeros([len(regions), len(regions)], dtype=int), index=regions, columns=regions)

for i in range(len(regions)):

    for j in range(len(regions)):

        corr.iloc[i, j] = len(set(s[i]) & set(s[j]))



sns.heatmap(corr, cmap='RdYlGn', center=50, annot=True, annot_kws={'size':14}, linewidths=6,

            cbar=False, square=True)



ax.set_title('The Number of Matching Games In Top 100 Between Regions\n', fontsize=16)

ax.xaxis.set_ticks_position('top')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14, verticalalignment='center')



plt.show()       
# Create a new dataframe with columns of 'Top_100_Sales', 'Grand_Total_Sales', 'Ratio' for each region

sales_100 = pd.DataFrame(index=regions, columns=['Top_100_Sales', 'Grand_Total_Sales', 'Ratio'])

for i, region in enumerate(regions):

    sales_100.iloc[i, 0] = data_100[i][region].sum()

    sales_100.iloc[i, 1] = data_analy[region].sum()

    

sales_100['Ratio'] = sales_100['Top_100_Sales'] / sales_100['Grand_Total_Sales']

sales_100
## Preparing data

global_100 = data_analy[:100] # The original dataset is ranked by Global_Sales



# Data for plotting on periods

sales_period = data_analy.groupby('Period')['Global_Sales'].sum()

counts_100_period = global_100.groupby('Period').size()



# Data for plotting on genre

sales_genre = data_analy.groupby('Genre')['Global_Sales'].sum()

sales_genre = sales_genre.sort_values(ascending=False)

counts_100_genre = global_100.groupby('Genre').size()

counts_100_genre = counts_100_genre.reindex(sales_genre.index) # unifying the index

counts_100_genre.fillna(0, inplace=True)

counts_100_genre = counts_100_genre.astype(int)



# Data for plotting on platforms

sales_platform = data_analy.groupby('Platform')['Global_Sales'].sum()

sales_platform = sales_platform.sort_values(ascending=False)

counts_100_platform = global_100.groupby('Platform').size()

counts_100_platform = counts_100_platform.reindex(sales_platform.index) # unifying the index

counts_100_platform.fillna(0, inplace=True)

counts_100_platform = counts_100_platform.astype(int)



# Plot(1)

fig, ax1 = plt.subplots(figsize=(10,6))



# Plot total sales of each period as background

ax1.bar(sales_period.index.values, sales_period, alpha=.5)

ax1.tick_params(axis='both', labelsize=14)

ax1.set_ylabel('Sales Revenue (US$ in Milli)', fontsize=14)



# Plot the number of games(top 100) within each period

ax2 = ax1.twinx()

x2 = counts_100_period.index.values

y2 = counts_100_period



ax2.plot(x2, y2, marker='o', linewidth=2.5, color='firebrick', alpha=.8)

ax2.set(ylim=(0, 100))

ax2.yaxis.set_tick_params(labelsize=14, labelcolor='firebrick')

ax2.set_ylabel('Number of Games', fontsize=14, color='firebrick')



for tx, ty in zip(x2, y2):

        ax2.text(tx, ty+1, ty, horizontalalignment='center', verticalalignment='bottom',

                 fontdict={'style':'italic', 'color':'firebrick', 'size':12})



plt.title('Top 100 Games Spread Across Periods', fontsize=20)





# Plot(2)

fig, ax1 = plt.subplots(figsize=(10,6))



# Plot total sales of each genre as background

ax1.bar(sales_genre.index.values, sales_genre, alpha=.5)

ax1.tick_params(axis='both', labelsize=14)

ax1.xaxis.set_tick_params(labelrotation=60)

ax1.set_ylabel('Sales Revenue (US$ in Milli)', fontsize=14)



# Plot the number of games(top 100) within each genre

ax2 = ax1.twinx()

x2 = counts_100_genre.index.values

y2 = counts_100_genre



ax2.plot(x2, y2, marker='o', linewidth=2.5, color='firebrick', alpha=.8)

ax2.set(ylim=(0, 100))

ax2.yaxis.set_tick_params(labelsize=14, labelcolor='firebrick')

ax2.set_ylabel('Number of Games', fontsize=14, color='firebrick')



for tx, ty in zip(x2, y2):

        ax2.text(tx, ty+1, ty, horizontalalignment='center', verticalalignment='bottom',

                 fontdict={'style':'italic', 'color':'firebrick', 'size':12})



plt.title('Top 100 Games Spread Across Genre', fontsize=20)



# Plot(3)

fig, ax1 = plt.subplots(figsize=(10,6))



# Plot total sales of each platform as background

ax1.bar(sales_platform.index.values, sales_platform, alpha=.5)

ax1.xaxis.set_tick_params(labelsize=10, labelrotation=60)

ax1.yaxis.set_tick_params(labelsize=14)

ax1.set_ylabel('Sales Revenue (US$ in Milli)', fontsize=14)



# Plot the number of games(top 100) within each platform

ax2 = ax1.twinx()

x2 = counts_100_platform.index.values

y2 = counts_100_platform



ax2.plot(x2, y2, marker='o', linewidth=2.5, color='firebrick', alpha=.8)

ax2.set(ylim=(0, 100))

ax2.yaxis.set_tick_params(labelsize=14, labelcolor='firebrick')

ax2.set_ylabel('Number of Games', fontsize=14, color='firebrick')



for tx, ty in zip(x2, y2):

        ax2.text(tx, ty+1, ty, horizontalalignment='center', verticalalignment='bottom',

                 fontdict={'style':'italic', 'color':'firebrick', 'size':12})



plt.title('Top 100 Games Spread Across Platforms', fontsize=20)



plt.show()
# Prepare data

publisher_sales = data_analy.groupby('Publisher')['Global_Sales'].sum()

publisher_sales.sort_values(ascending=False, inplace=True)

publisher_top_10 = publisher_sales[:10]

publisher_others = pd.Series(publisher_sales[10:].sum(), index=['All_Others'])

publisher_top_10 = publisher_top_10.append(publisher_others)

publisher_top_10.sort_values(ascending=False, inplace=True)



# Plot

plt.figure(figsize=(10,8))

explode = (0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)

plt.pie(publisher_top_10, labels=publisher_top_10.index.values, autopct='%.1f%%', pctdistance=.8,

        startangle=60, textprops=dict(size=12, weight='bold'), explode=explode)



plt.title('The Proportion of Top 10 Publishers', fontsize=20)

plt.show()