import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
path="../input/vgsales.csv"



data = pd.read_csv("../input/videogamesales/vgsales.csv",header=0)



data.head(5)
data.isna().sum()
data.drop(data.index[data.Year.isna()],inplace=True,axis=0)

data.drop(data.index[data.Publisher.isna()],inplace=True,axis=0)

data.info()
data.describe()
# Draw Plot

plt.figure(figsize=(25,8), dpi= 80)

sns.kdeplot(data.Global_Sales, shade=True, color="g", label="NA_Sales", alpha=.7)



# Decoration

plt.title('Global Sales Distribution', fontsize=16)

plt.legend()

plt.show()
#top platforms (name of the platform,total number of games developed for that platform)

topPlatforms_index = data.Platform.value_counts().head(10).index

topPlatforms_values = data.Platform.value_counts().head(10).values



#top genres (name of the genre,total number of games developed in that genre)

topGenres_index = data.Genre.value_counts().head(10).index

topGenres_values = data.Genre.value_counts().head(10).values



#top game developers/publishers (name of the publisher,total number of games published by that publisher)

topPublisher_index = data.Publisher.value_counts().head(10).index

topPublisher_values = data.Publisher.value_counts().head(10).values



fig, (ax1,ax2) = plt.subplots(1,2,figsize=(25,8), facecolor='white')



##top platforms used for games

ax1.vlines(x=topPlatforms_index, ymin=0, ymax=topPlatforms_values, color='#0D59D5', linewidth=30)

ax1.set_title('Top 10 Platforms',fontsize=16)



#top genres is which the games are developed

ax2.vlines(x=topGenres_index, ymin=0, ymax=topGenres_values, color='#AB0DD5', linewidth=30)

ax2.set_title('Top 10 Genres',fontsize=16)

plt.show()



fig, ax = plt.subplots(figsize=(25,8), facecolor='white')



#top publishers of the games

ax.vlines(x=topPublisher_index, ymin=0, ymax=topPublisher_values, color='#D60F79', linewidth=30)

ax.set_title('Top 10 Publishers',fontsize=16)

# Basic correlogram

sns.pairplot(data.loc[0:,['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']])

plt.show()

def yearly_Sales(Genres):

    for i in Genres:

        genre_sales = data[data['Genre'] == i]

        plt.figure(figsize=(10,8),dpi=80)

        genre_sales[['Year','Global_Sales']].groupby('Year')['Global_Sales'].sum().plot(label=i)

        plt.legend()

        plt.title(i)   

yearly_Sales(['Sports'])        

        