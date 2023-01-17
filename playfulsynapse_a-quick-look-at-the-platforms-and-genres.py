import pandas as pd

import numpy as np

import seaborn as sns; sns.set()



import matplotlib.pyplot as plt

%matplotlib inline 



from matplotlib_venn import venn3 



#ignore some anoying warnings from the seaborn plot

import warnings

warnings.filterwarnings('ignore')
# Import the data

gamesDB=pd.read_csv('../input/ign.csv')





# Print an overview of the data set

print("number of rows: ",gamesDB.shape[0])

gamesDB.head()
# Remove genres with less than 500 titles.

genres=gamesDB.groupby('genre')['genre']

genres_count=genres.count()

large_genres=genres_count[genres_count>=500]

large_genres.sort_values(ascending=False,inplace=True)

print("number of genres with at least 500 titles: ",large_genres.shape[0] )

print(large_genres)

genre_list=large_genres.index.values



#Extract only the genres with more than 500 titles.

gamesDB_large_genre=gamesDB[gamesDB['genre'].isin(genre_list)]



#Use pandas pivot table to aggregate the number of releases by year

table_genre_by_year = pd.pivot_table(gamesDB_large_genre,values=['score'],index=['release_year'],columns=['genre'],aggfunc='count',margins=False)

table_genre_by_year['score'].plot(kind='bar', stacked=True,figsize=(12,7),colormap='Accent')
# Remove platforms with less than 500 titles.

platforms=gamesDB.groupby('platform')['platform']

platforms_count=platforms.count()

large_platforms=platforms_count[platforms_count>=500]

large_platforms.sort_values(ascending=False,inplace=True)

print("number of platforms with at least 500 titles: ",large_platforms.shape[0] )

print(large_platforms)
platform_list=large_platforms.index.values



#Extract only the platforms with more than 500 titles.

gamesDB_large_platform=gamesDB[gamesDB['platform'].isin(platform_list)]



#Use pandas pivot table to aggregate the number of releases by year

table_platform_by_year = pd.pivot_table(gamesDB_large_platform,values=['score'],index=['release_year'],columns=['platform'],aggfunc='count',margins=False)

table_platform_by_year['score'].plot(kind='bar', stacked=True,figsize=(12,7),colormap='Accent')

# Venn diagram of selected rows

all_titles_set=set(gamesDB['Unnamed: 0'].values.flatten())

platform_set=set(gamesDB_large_platform['Unnamed: 0'].values.flatten())

genre_set=set(gamesDB_large_genre['Unnamed: 0'].values.flatten())



venn3([all_titles_set, platform_set, genre_set], ('All titles', 'large platforms', 'large genres'))



plt.show()
#screen for both large genre and platform

gamesDB_large=gamesDB[gamesDB['genre'].isin(genre_list)]

gamesDB_large=gamesDB_large[gamesDB_large['platform'].isin(platform_list)]



#create pivot table of the number of reviews for each platform and genre combination

table_count = pd.pivot_table(gamesDB_large,values=['score'],index=['platform'],columns=['genre'],aggfunc='count',margins=False)



sns.set_context("talk") # make the table a bit bigger than the default



# use the table_count['score'], since the pivot_table returns a multiindex 

# that does not look nice when drawn in the heat map 

ax=sns.heatmap(table_count['score'], linewidths=.5,annot=True, fmt="d")
#create pivot table of the average review score for each platform and genre combination

table_avg_score = pd.pivot_table(gamesDB_large,values=['score'],index=['platform'],columns=['genre'],aggfunc=np.average,margins=True)

cmap=sns.diverging_palette(10, 220, sep=80, n=7, as_cmap=True)

sns.set_context("talk") # make the table a bit bigger than the default



ax=sns.heatmap(table_avg_score['score'], linewidths=.5,annot=True,cmap=cmap)